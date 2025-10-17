#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cctype>
#include <algorithm>
#include <math.h>


const int NUM_THREADS = 4; //Num Threads

// Sobel Filter Kernel
static float SOBEL_X_KERNEL[9] = { -1,0,1, -2,0,2, -1,0,1 };
static float SOBEL_Y_KERNEL[9] = { -1,-2,-1, 0,0,0, 1,2,1 };

// All great things start with an FSM 
enum class ComputeStages: int {GRAY_SCALE = 0, SOBEL_X = 1, SOBEL_Y  = 2, MAGNITUDE = 3, STOP = 4}; 

// Shared data between main thread and worker threads
struct FrameJob {
    const cv::Mat* bgrFrame = nullptr; 
    cv::Mat* grayFrame = nullptr; 
    cv::Mat* sobelXOut = nullptr; 
    cv::Mat* sobelYOut = nullptr; 
    cv::Mat* mag32fOut = nullptr; 
    int imgRows = 0, imgCols= 0; 

    std::atomic<ComputeStages> currentStage{ComputeStages::STOP}; //Shared Resource, so atomic required. 

    pthread_barrier_t startBarrier; 
    pthread_barrier_t endBarrier;

};

struct WorkerContext { 
    FrameJob* job = nullptr; 
    int tid = 0; 
};

//Calculates the magnitude of GX and GY
static inline void magnitude(const cv::Mat& gx32f, const cv::Mat& gy32f, cv::Mat& mag32f, int yStart, int yEnd){
    const int rows = gx32f.rows,  cols = gx32f.cols;
    const int ys = std::max(0, yStart); // Starting row, max to prevent clipping
    const int ye = std::min(rows, yEnd); // Ending row, min to prevent clipping

    for(int y = ys; y < ye; ++y){
        const float* gX_row = gx32f.ptr<float>(y);
        const float* gY_row = gy32f.ptr<float>(y); 
        float* out = mag32f.ptr<float>(y); 
        for (int x = 0; x < cols; ++x) {
            out[x] = sqrt(gX_row[x] * gX_row[x] + gY_row[x] * gY_row[x]);
        }
    }
}

static inline void convolve3x3_rowptr(const cv::Mat& src, cv::Mat& dst,
                                      const float kernel[9], int yStart, int yEnd){
    const int border = 1;
    const int rows = src.rows, cols = src.cols;
    const int ys = std::max(border, yStart); // Starting row, max to prevent clipping
    const int ye = std::min(rows - border, yEnd); // Ending row, min to prevent clipping

    for (int y = ys; y < ye; ++y) {
        // Store pointers to prev, curr, and next row
        const uchar* prevRow = src.ptr<uchar>(y - 1);
        const uchar* currRow = src.ptr<uchar>(y);
        const uchar* nextRow = src.ptr<uchar>(y + 1);
        float* outRow = dst.ptr<float>(y); //Pointer to output row.

        //Convolve entire row with specified kernel 
        for (int x = border; x < cols - border; ++x) {
            float sum =
                prevRow[x-1]*kernel[0] + prevRow[x]*kernel[1] + prevRow[x+1]*kernel[2] +
                currRow[x-1]*kernel[3] + currRow[x]*kernel[4] + currRow[x+1]*kernel[5] +
                nextRow[x-1]*kernel[6] + nextRow[x]*kernel[7] + nextRow[x+1]*kernel[8];
            outRow[x] = sum;
        }
    }
}

static inline void grayscale_calculation(const cv::Mat& bgr, cv::Mat& gray,int yStart, int yEnd){
    const int rows = bgr.rows, cols = bgr.cols;
    const int ys = std::max(0, yStart);
    const int ye = std::min(rows, yEnd);

    for (int y = ys; y < ye; ++y) {
        const cv::Vec3b* in = bgr.ptr<cv::Vec3b>(y);
        uchar* out = gray.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            const uchar B = in[x][0], G = in[x][1], R = in[x][2];
            out[x] = static_cast<uchar>((R*77 + G*150 + B*29 + 128) >> 8);
        }
    }
}


void* workerThread(void* arg){
    WorkerContext* ctx = static_cast<WorkerContext*>(arg);
    FrameJob* job = ctx->job;

    for (;;) {
        // Wait for main thread to assign a new stage
        pthread_barrier_wait(&job->startBarrier);

        ComputeStages stage = job->currentStage.load(std::memory_order_acquire);
        if (stage == ComputeStages::STOP) break;

        // Compute my portion of rows for this stage
        int totalRows = job->imgRows;
        int rowsPerThread = (totalRows + NUM_THREADS - 1) / NUM_THREADS;
        int yStart = ctx->tid * rowsPerThread;
        int yEnd = std::min(totalRows, yStart + rowsPerThread);

        if (stage == ComputeStages::SOBEL_X) {
            convolve3x3_rowptr(*job->grayFrame, *job->sobelXOut, SOBEL_X_KERNEL, yStart, yEnd);
        } else if (stage == ComputeStages::SOBEL_Y) {
            convolve3x3_rowptr(*job->grayFrame, *job->sobelYOut, SOBEL_Y_KERNEL, yStart, yEnd);
        } else if (stage == ComputeStages::GRAY_SCALE){
            grayscale_calculation(*job->bgrFrame,*job->grayFrame,yStart, yEnd);
        } else if (stage == ComputeStages::MAGNITUDE){
            magnitude(*job->sobelXOut, *job->sobelYOut, *job->mag32fOut ,yStart, yEnd);
        }

        // Signal that this thread has finished its slice
        pthread_barrier_wait(&job->endBarrier);
    }

    // Final sync so main can safely join threads
    pthread_barrier_wait(&job->endBarrier);
    return nullptr;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " video_path or camera index>\n";
        return 1; 
    }

    cv::VideoCapture cap; 
    if(std::isdigit(argv[1][0])) cap.open(std::stoi(argv[1]));
    else cap.open(argv[1]);
    if(!cap.isOpened()){
        std::cerr << "Error: could not open source.\n";
        return 1; 
    }

    FrameJob job; 
    pthread_barrier_init(&job.startBarrier,nullptr,NUM_THREADS + 1);
    pthread_barrier_init(&job.endBarrier, nullptr, NUM_THREADS + 1); 

    pthread_t threads[NUM_THREADS]; 
    WorkerContext workerCtx[NUM_THREADS]; 
    for (int i = 0; i < NUM_THREADS; ++i){
        workerCtx[i] = WorkerContext{&job, i }; 
        pthread_create(&threads[i], nullptr, workerThread, &workerCtx[i]);
    }

    cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

    //Init for grayscale and sobel
    cv::Mat frame, gray, gX32f, gY32f, mag32f, mag8u; 

    while(1){
        if (!cap.read(frame) || frame.empty()) break;

        // (Re)allocate outputs on size change
        if (gray.size() != frame.size()) {
            gray.create(frame.size(), CV_8UC1);
            gX32f.create(frame.size(), CV_32FC1);
            gY32f.create(frame.size(), CV_32FC1);
            mag32f.create(frame.size(), CV_32FC1);
        }
        gX32f.setTo(0);
        gY32f.setTo(0);
    
        // Share current frame info
        job.bgrFrame  = &frame;
        job.grayFrame = &gray;
        job.sobelXOut = &gX32f;
        job.sobelYOut = &gY32f;
        job.mag32fOut = &mag32f; 
        job.imgRows   = gray.rows;
        job.imgCols   = gray.cols;
    
        // ---- Stage 1: Parallel GRAYSCALE ----
        job.currentStage.store(ComputeStages::GRAY_SCALE, std::memory_order_release); //Thread safe stage changing
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);
    
        // ---- Stage 2: Parallel SOBEL X ----
        job.currentStage.store(ComputeStages::SOBEL_X, std::memory_order_release); //Thread safestage changing
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);
    
        // ---- Stage 3: Parallel SOBEL Y ----
        job.currentStage.store(ComputeStages::SOBEL_Y, std::memory_order_release); //Thread safe stage changing
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);
        
        // ---- Stage 4: Parallel Magnitude ---
        job.currentStage.store(ComputeStages::MAGNITUDE, std::memory_order_release); //Thread safe stage changing
        pthread_barrier_wait(&job.startBarrier);
        pthread_barrier_wait(&job.endBarrier);

        // Combine results & display
        mag32f.convertTo(mag8u, CV_8U, 0.5);
        cv::imshow("Sobel", mag8u);

        // Close Window
        int key = cv::waitKey(1);             
        if (key == 27) break;                    // ESC to quit
        if (cv::getWindowProperty("Sobel", cv::WND_PROP_VISIBLE) < 1) break;
    }

    // Tell workers to stop
    job.currentStage.store(ComputeStages::STOP, std::memory_order_release);
    pthread_barrier_wait(&job.startBarrier);
    pthread_barrier_wait(&job.endBarrier);

    for (int i = 0; i < NUM_THREADS; ++i) pthread_join(threads[i], nullptr);
    pthread_barrier_destroy(&job.startBarrier);
    pthread_barrier_destroy(&job.endBarrier);
    return 0;
}



