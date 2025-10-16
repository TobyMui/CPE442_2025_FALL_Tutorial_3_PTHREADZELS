// sobel_barriers.cpp
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cctype>
#include <algorithm>

static const int NUM_THREADS = 4;

enum class Pass : int { SOBEL_X = 0, SOBEL_Y = 1, STOP = 2 };

struct Shared {
    const cv::Mat* gray = nullptr;
    cv::Mat* gx = nullptr;
    cv::Mat* gy = nullptr;
    int rows = 0, cols = 0;

    std::atomic<Pass> pass{Pass::STOP};

    // two rendezvous points: start (kick off work) and end (work finished)
    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;
};

struct ThreadCtx {
    Shared* S = nullptr;
    int tid = 0; // 0..NUM_THREADS-1
};

static inline void convolve3x3_rowptr(const cv::Mat& src, cv::Mat& dst,
                                      const float K[9], int y0, int y1)
{
    const int offset = 1;
    const int rows = src.rows, cols = src.cols;
    const int ys = std::max(offset, y0);
    const int ye = std::min(rows - offset, y1);

    for (int y = ys; y < ye; ++y) {
        const uchar* ym1 = src.ptr<uchar>(y - 1);
        const uchar* y0p = src.ptr<uchar>(y);
        const uchar* yp1 = src.ptr<uchar>(y + 1);
        float* out = dst.ptr<float>(y);

        for (int x = offset; x < cols - offset; ++x) {
            float sum =
                ym1[x-1]*K[0] + ym1[x]*K[1] + ym1[x+1]*K[2] +
                y0p[x-1]*K[3] + y0p[x]*K[4] + y0p[x+1]*K[5] +
                yp1[x-1]*K[6] + yp1[x]*K[7] + yp1[x+1]*K[8];
            out[x] = sum;
        }
    }
}

static float SOBEL_X[9] = { -1,0,1, -2,0,2, -1,0,1 };
static float SOBEL_Y[9] = { -1,-2,-1, 0,0,0, 1,2,1 };

void* worker(void* arg)
{
    ThreadCtx* C = static_cast<ThreadCtx*>(arg);
    Shared* S = C->S;

    for (;;) {
        // 1) Wait for main to announce a pass (or STOP)
        pthread_barrier_wait(&S->start_barrier);

        Pass p = S->pass.load(std::memory_order_acquire);
        if (p == Pass::STOP) break;

        // 2) Compute my slice for the announced pass
        int rows = S->rows, cols = S->cols;
        int rows_per = (rows + NUM_THREADS - 1) / NUM_THREADS;
        int y0 = C->tid * rows_per;
        int y1 = std::min(rows, y0 + rows_per);

        if (p == Pass::SOBEL_X) {
            convolve3x3_rowptr(*S->gray, *S->gx, SOBEL_X, y0, y1);
        } else { // Pass::SOBEL_Y
            convolve3x3_rowptr(*S->gray, *S->gy, SOBEL_Y, y0, y1);
        }

        // 3) Signal this pass is finished
        pthread_barrier_wait(&S->end_barrier);
    }

    // final rendezvous to let main join cleanly
    pthread_barrier_wait(&S->end_barrier);
    return nullptr;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path or camera index>\n";
        return 1;
    }

    cv::VideoCapture cap;
    if (std::isdigit(argv[1][0])) cap.open(std::stoi(argv[1]));
    else cap.open(argv[1]);

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open source.\n";
        return 1;
    }

    // We manage our own threading -> avoid oversubscription
    cv::setNumThreads(1);

    Shared S;
    pthread_barrier_init(&S.start_barrier, nullptr, NUM_THREADS + 1); // +1 for main
    pthread_barrier_init(&S.end_barrier,   nullptr, NUM_THREADS + 1);

    pthread_t threads[NUM_THREADS];
    ThreadCtx ctx[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        ctx[i] = ThreadCtx{ &S, i };
        pthread_create(&threads[i], nullptr, worker, &ctx[i]);
    }

    cv::namedWindow("Gray",  cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, gray, gx32f, gy32f, mag32f, mag8u;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (gx32f.size() != gray.size()) {
            gx32f.create(gray.size(), CV_32FC1);
            gy32f.create(gray.size(), CV_32FC1);
            mag32f.create(gray.size(), CV_32FC1);
        }
        gx32f.setTo(0);
        gy32f.setTo(0);

        // publish shared pointers/dims for this frame
        S.gray = &gray;
        S.gx   = &gx32f;
        S.gy   = &gy32f;
        S.rows = gray.rows;
        S.cols = gray.cols;

        // ---- Pass 1: Gx ----
        S.pass.store(Pass::SOBEL_X, std::memory_order_release);
        pthread_barrier_wait(&S.start_barrier); // release workers for X
        pthread_barrier_wait(&S.end_barrier);   // wait until X done

        // ---- Pass 2: Gy ----
        S.pass.store(Pass::SOBEL_Y, std::memory_order_release);
        pthread_barrier_wait(&S.start_barrier); // release workers for Y
        pthread_barrier_wait(&S.end_barrier);   // wait until Y done

        // Now both Gx & Gy complete → compute magnitude & display on main
        cv::magnitude(gx32f, gy32f, mag32f);
        // fixed gain (avoids per-frame pumping). Tune as desired.
        mag32f.convertTo(mag8u, CV_8U, 0.5);

        cv::imshow("Gray", gray);
        cv::imshow("Sobel", mag8u);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (cv::getWindowProperty("Gray", cv::WND_PROP_VISIBLE) < 1) break;
    }

    // stop workers and join
    S.pass.store(Pass::STOP, std::memory_order_release);
    pthread_barrier_wait(&S.start_barrier); // wake for STOP
    pthread_barrier_wait(&S.end_barrier);   // let them exit

    for (int i = 0; i < NUM_THREADS; ++i) pthread_join(threads[i], nullptr);
    pthread_barrier_destroy(&S.start_barrier);
    pthread_barrier_destroy(&S.end_barrier);
    return 0;
}
