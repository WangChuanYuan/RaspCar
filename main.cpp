#define _DEBUG

#include <cstdlib>
#include <iostream>
#include <vector>
#include <windows.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "GPIOlib.h"

#define PI 3.14159265

using namespace cv;
using namespace std;
using namespace GPIO;

//const string CAM_PATH = "/dev/video0";
const string CAM_PATH = "0";
const string MAIN_WINDOW_NAME = "View";
const string CANNY_WINDOW_NAME = "Canny";

const int CANNY_LOWER_BOUND = 50;
const int CANNY_UPPER_BOUND = 250;
const int HOUGH_THRESHOLD = 150;

typedef struct PID {
    int set_speed;
    int error;
    int error_next;
    int error_last;
    float kp, ki, kd;
} PID;

#define P_DATA 0.2
#define I_DATA 0.01
#define D_DATA 0.2
#define SET_SPEED 100
#define HAVE_NEW_VELOCITY 0x01

static PID sPID;
static PID *sptr = &sPID;
int leftSpeed;
int rightSpeed;
int flag;

void readSpeed() {
    resetCounter();
    delay(1000);
    getCounter(&leftSpeed, &rightSpeed);
    flag |= HAVE_NEW_VELOCITY;
}

void incPIDInit() {
    sptr->set_speed = SET_SPEED;
    sptr->error = 0;
    sptr->error_next = 0;
    sptr->error_last = 0;
    sptr->kp = P_DATA;
    sptr->ki = I_DATA;
    sptr->kd = D_DATA;
}

int incPIDCalc(int actual_speed) {
    sptr->error = sptr->set_speed - actual_speed;
    float inc_speed, res_speed;
    inc_speed = sptr->kp * (sptr->error - sptr->error_next) +
                sptr->ki * sptr->error +
                sptr->kd * (sptr->error - 2 * sptr->error_next + sptr->error_last);
    res_speed = actual_speed + inc_speed;
    sptr->error_last = sptr->error_next;
    sptr->error_next = sptr->error;
    return (int) res_speed;
}

int main() {
    //init
    init();
    incPIDInit();
    leftSpeed = 0;
    rightSpeed = 0;
    flag = 0;

    VideoCapture capture(CAM_PATH);
    //If this fails, try to open as a video camera, through the use of an integer param
    if (!capture.isOpened()) {
        capture.open(atoi(CAM_PATH.c_str()));
    }
    //The time for camera's initialization
    Sleep(1000);

    Mat image;
    while (true) {

        readSpeed();
        if (flag & HAVE_NEW_VELOCITY) {
            controlLeft(FORWARD, incPIDCalc(leftSpeed));
            controlRight(FORWARD, incPIDCalc(rightSpeed));
            flag &= ~HAVE_NEW_VELOCITY;
        }

        capture >> image;
        if (image.empty())
            break;
        //Set the ROI(Region Of Interest) for the image
        Rect roi(0, image.rows / 3, image.cols, image.rows / 3);
        Mat imgROI = image(roi);

        //Canny algorithm
        Mat contours;
        Canny(imgROI, contours, CANNY_LOWER_BOUND, CANNY_UPPER_BOUND);
#ifdef _DEBUG
        imshow(CANNY_WINDOW_NAME, contours);
#endif

        vector<Vec2f> lines;
        HoughLines(contours, lines, 1, PI / 180, HOUGH_THRESHOLD);
        Mat result(imgROI.size(), CV_8U, Scalar(255));
        imgROI.copyTo(result);

        //find out the left and right boundaries
        Vec2f leftBound, rightBound;
        bool hasLeft, hasRight;
        float minLeft = result.cols / 2.0;
        float minRight = result.cols / 2.0;
        for (vector<Vec2f>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
            float rho = (*it)[0];            //First element is distance rho
            float theta = (*it)[1];        //Second element is angle theta

            //Filter to find out lines left and right, and atan(0.09) equals about 5 degrees
            if (theta > 0.09 && theta < 1.48 && rho < minRight){
                hasRight = true;
                minRight = rho;
                rightBound = *it;
            }
            else if (theta > 1.62 && theta < 3.05 && rho < minLeft){
                hasLeft = true;
                minLeft = rho;
                leftBound = *it;
            }
        }

#ifdef _DEBUG
        if(hasLeft){
            //point of intersection of the line with first row
            Point pt1(leftBound[0] / cos(leftBound[1]), 0);
            //point of intersection of the line with last row
            Point pt2((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 3, CV_AA);
        }
        if(hasRight){
            //point of intersection of the line with first row
            Point pt1(rightBound[0] / cos(rightBound[1]), 0);
            //point of intersection of the line with last row
            Point pt2((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);
            //Draw a line
            line(result, pt1, pt2, Scalar(0, 255, 255), 3, CV_AA);
        }
        imshow(MAIN_WINDOW_NAME, result);
#endif

        //decide directions

        lines.clear();
        waitKey(1);
    }
    return 0;
}