#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h> 

#define NUM_THREADS 4

pthreads_t threads[4]; 

 // Declare Sobel Kernels (X and Y)
 cv::Mat sobelX = (cv::Mat_<float>(3,3) <<
 -1, 0, 1,
 -2, 0, 2,
 -1, 0, 1);

cv::Mat sobelY = (cv::Mat_<float>(3,3) <<
 -1, -2, -1,
  0,  0,  0,
  1,  2,  1);


cv::Mat convolution(const cv::Mat& gray, const cv::Mat& kernel){
    CV_Assert(gray.channels() == 1);
    cv::Mat result(gray.rows, gray.cols, CV_32F, cv::Scalar(0));

    //Offset 
    int offset = 1; 

    //Double for loop for traversing the 2d Matrix
    for (int y = offset; y < gray.rows - offset; y++){
        for(int x = offset; x < gray.cols - offset; x++){
            float sum = 0.0f;

            //Apply the kernel 
            for(int ky = 0; ky <= 2; ky++){
                for(int kx = 0; kx <= 2; kx++){
                    float pixel = static_cast<float>(gray.at<uchar>(y + ky, x + kx));
                    float weight = kernel.at<float>(ky, kx); 
                    sum += pixel * weight; 
                }
            }
            result.at<float>(y,x) = sum; 
        }
    }
    return result;
}

int main(int argc, char** argv) {

    std::cout << "Press esc or q to close the window! \n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }



    const std::string kWin = "Grayscale Video";

    cv::VideoCapture cap(argv[1]);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::namedWindow(kWin, cv::WINDOW_AUTOSIZE);
    cv::Mat frame, gray;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "End of video." << std::endl;
            break;
        }

        // Manually convert to grayscale using CCIR 601 coefficients
        gray.create(frame.rows, frame.cols, CV_8UC1);
        for (int y = 0; y < frame.rows; y++) {
            for (int x = 0; x < frame.cols; x++) {
                cv::Vec3b color = frame.at<cv::Vec3b>(y, x);
                uchar B = color[0];
                uchar G = color[1];
                uchar R = color[2];

                uchar Y = static_cast<uchar>(0.299 * R + 0.587 * G + 0.114 * B);
                gray.at<uchar>(y, x) = Y;
            }
        }

        //Initialize X and Y output matrices
        cv::Mat gx32f, gy32f;

        //Apply the respective kernels to X and Y output matrices
        gx32f = convolution(gray,sobelX);
        gy32f = convolution(gray,sobelY);

        // Calculate the Magnitude 
        cv::Mat mag32f;
        cv::magnitude(gx32f, gy32f, mag32f);

        // Convert for display
        cv::Mat gx8u, gy8u, mag8u;
        cv::convertScaleAbs(gx32f, gx8u);                 // |Gx|
        cv::convertScaleAbs(gy32f, gy8u);                 // |Gy|
        cv::normalize(mag32f, mag32f, 0, 255, cv::NORM_MINMAX);
        mag32f.convertTo(mag8u, CV_8U);

        // Show windows
        cv::imshow(kWin, gray);
        cv::imshow("Sobel Filter", mag8u);

        // Handle keys and window events
        int key = cv::waitKey(25);
        if (key == 'q' || key == 27) { // quit on 'q' or ESC
            break;
        }
        if (cv::getWindowProperty(kWin, cv::WND_PROP_AUTOSIZE) < 0) {
            break; // window closed
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
