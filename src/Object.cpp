#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double calculateDistance(double disparity, double focalLength, double baseline) {
    if (disparity == 0) return -1;
    return (focalLength * baseline) / disparity;
}

int main() {
    VideoCapture capLeft(0), capRight(1);
    if (!capLeft.isOpened() || !capRight.isOpened()) {
        cout << "Error: Could not open one or both cameras." << endl;
        return -1;
    }

    double focalLength = 800.0;
    double baseline = 10.0;

    Ptr<StereoSGBM> stereo = StereoSGBM::create(
        0,
        16,
        3,
        8 * 3 * 3,
        32 * 3 * 3,
        1,
        63,
        10,
        100,
        32,
        StereoSGBM::MODE_SGBM
    );

    Mat frameLeft, frameRight, grayLeft, grayRight, disparity, disparityNorm;
    while (true) {
        capLeft >> frameLeft;
        capRight >> frameRight;
        if (frameLeft.empty() || frameRight.empty()) {
            cout << "Error: No frame captured from one or both cameras." << endl;
            break;
        }

        cvtColor(frameLeft, grayLeft, COLOR_BGR2GRAY);
        cvtColor(frameRight, grayRight, COLOR_BGR2GRAY);

        stereo->compute(grayLeft, grayRight, disparity);

        normalize(disparity, disparityNorm, 0, 255, NORM_MINMAX, CV_8U);

        int centerX = frameLeft.cols / 2;
        int centerY = frameLeft.rows / 2;
        double disparityValue = disparity.at<short>(centerY, centerX) / 16.0;
        double distance = calculateDistance(disparityValue, focalLength, baseline);

        string text = "Distance: " + (distance > 0 ? to_string(distance) : "N/A") + " cm";
        putText(frameLeft, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);

        imshow("Left Camera", frameLeft);
        imshow("Right Camera", frameRight);
        imshow("Disparity Map", disparityNorm);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    capLeft.release();
    capRight.release();
    destroyAllWindows();
    return 0;
}