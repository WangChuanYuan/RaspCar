#define _DEBUG
#define _VIDEO

#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "GPIOlib.h"

#define E 0.1
#define SET_SPEED 7

using namespace cv;
using namespace std;
using namespace GPIO;

const string CAM_PATH = "/dev/video0";

const int CANNY_LOWER_BOUND = 50;
const int CANNY_UPPER_BOUND = 250;
const int HOUGH_THRESHOLD = 80;

const int leftAngle = -18;
const int lleftAngle = -23;
const int rightAngle = 5;
const int lrightAngle = 8;

bool hasObstacle = false;
bool obstacleL = false;
bool obstacleR = false;

const Scalar hsvRedLo(0, 70, 50);
const Scalar hsvRedHi(10, 255, 255);

int noLinesCount;

double distance(Point2f point, double k, double b);

void initialize();

void forward();

void stop();

int main() {
    initialize();

#ifdef _VIDEO
    VideoCapture capture(CAM_PATH);
    if (!capture.isOpened()) {
        capture.open(atoi(CAM_PATH.c_str()));
    }
#endif

    Mat image = imread("view.jpg");
    while (true) {

#ifdef _VIDEO
        capture >> image;
#endif
        if (image.empty()) {
            break;
        }

        Rect roi(0, image.rows / 2, 4 * image.cols / 5, 2 * image.rows / 4);
        Mat imgROI = image(roi);
        Mat result(imgROI.size(), CV_8U, Scalar(255));
        imgROI.copyTo(result);

        // find red obstacle
        Mat imgHSV;
        cvtColor(imgROI, imgHSV, CV_BGR2HSV);
        Mat maskRed;
        inRange(imgHSV, hsvRedLo, hsvRedHi, maskRed);
        vector<vector<Point>> boxes;
        vector<Vec4i> hierarchy;
        Point2f obstacle[4];
        double maxSize = (result.rows * result.cols) / 45.0;
        findContours(maskRed, boxes, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        bool flag = false;
        for (int i = 0; i < boxes.size(); i++) {
            RotatedRect box = minAreaRect(Mat(boxes[i]));
            if (box.size.area() > maxSize) {
                flag = true;
                maxSize = box.size.area();
                box.points(obstacle);
            }
        }
        hasObstacle = flag;
#ifdef _DEBUG
        if (hasObstacle) {
            for (int i = 0; i < 4; i++) {
                line(result, obstacle[i], obstacle[(i + 1) % 4], Scalar(0, 0, 255), 2, CV_AA);
            }
        }
        imshow("Red Area", maskRed);
#endif

        // find lines
        Mat imgGray;
        cvtColor(imgROI, imgGray, CV_BGR2GRAY);
        Mat imgDilate, dilateElement;
        dilateElement = getStructuringElement(MORPH_RECT, Size(1, 1));
        dilate(imgGray, imgDilate, dilateElement);
        Mat imgErode, erodeElement;
        erodeElement = getStructuringElement(MORPH_RECT, Size(1, 1));
        dilate(imgDilate, imgErode, erodeElement);
        Mat contours;
        Canny(imgErode, contours, CANNY_LOWER_BOUND, CANNY_UPPER_BOUND);
        vector<Vec2f> lines;
        HoughLines(contours, lines, 1, CV_PI / 180, HOUGH_THRESHOLD);
#ifdef _DEBUG
        imshow("Canny", contours);
#endif

        // find out the left and right boundaries
        Vec2f leftBound, rightBound;
        Point2f left1, left2;
        Point2f right1, right2;
        bool hasLeft = false, hasRight = false;
        Point2f center(result.cols / 2, result.rows / 2);
        double minLeft = result.cols / 2.0;
        double minRight = result.cols / 2.0;
        for (vector<Vec2f>::const_iterator it = lines.begin(); it != lines.end(); ++it) {
            float rho = (*it)[0];            //First element is distance rho
            float theta = (*it)[1];        //Second element is angle theta

            Point2f pt1(rho / cos(theta), 0);
            Point2f pt2((rho - result.rows * sin(theta)) / cos(theta), result.rows);

            //filter out the vertical or horizontal lines
            //if (!(theta > 0.09 && theta < 1.48) && !(theta > 1.62 && theta < 3.05))
            //continue;
            if (theta > 1.48 && theta < 1.62)
                continue;

            double k = (pt2.y - pt1.y) / (pt2.x - pt1.x);
            double b = pt2.y - k * pt2.x;

            double dist = distance(center, k, b);
            double limitK = 0.2;

            if (k < -limitK && dist < minLeft) {
                hasLeft = true;
                minLeft = dist;
                leftBound = *it;
            } else if (k > limitK && dist < minRight) {
                hasRight = true;
                minRight = dist;
                rightBound = *it;
            }

#ifdef _DEBUG
            line(result, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
#endif
        }

        if (hasLeft) {
            left1 = Point2f(leftBound[0] / cos(leftBound[1]), 0);
            left2 = Point2f((leftBound[0] - result.rows * sin(leftBound[1])) / cos(leftBound[1]), result.rows);
        }

        if (hasRight) {
            right1 = Point2f(rightBound[0] / cos(rightBound[1]), 0);
            right2 = Point2f((rightBound[0] - result.rows * sin(rightBound[1])) / cos(rightBound[1]), result.rows);
        }

        if (hasObstacle) {
            // decide the side of the obstacle
            Point2f obstacleCenter((obstacle[0].x + obstacle[2].x) / 2.0,
                                   (obstacle[0].y + obstacle[2].y) / 2.0);
            double target = sqrt(pow(obstacle[0].y - obstacle[2].y, 2) + pow(obstacle[0].x - obstacle[2].x, 2)) / 2;

            double kLeft, bLeft, distLeft;
            double kRight, bRight, distRight;
            if (hasLeft) {
                kLeft = (left2.y - left1.y) / (left2.x - left1.x);
                bLeft = left2.y - kLeft * left2.x;
                distLeft = distance(obstacleCenter, kLeft, bLeft);
            }
            if (hasRight) {
                kRight = (right2.y - right1.y) / (right2.x - right1.x);
                bRight = right2.y - kRight * right2.x;
                distRight = distance(obstacleCenter, kRight, bRight);
            }

            // 如果左右都未判断有障碍物则进行判断，否则保留上次判断结果
            if (!obstacleR && !obstacleL) {
                if (hasLeft && !hasRight) {
                    if (distLeft > 1.3 * target)
                        obstacleR = true;
                    else obstacleL = true;
                } else if (hasRight && !hasLeft) {
                    if (distRight > 1.3 * target)
                        obstacleL = true;
                    else obstacleR = true;
                } else if (hasLeft && hasRight) {
                    if (distLeft > distRight)
                        obstacleR = true;
                    else obstacleL = true;
                } else {
                    if (obstacleCenter.x < result.cols / 2)
                        obstacleL = true;
                    else
                        obstacleR = true;
                }
            }

            // let the obstacle be the boundary
            if (obstacleL) {
                hasLeft = true;
                left1 = obstacle[2];
                left2 = Point2f((obstacle[0].x + obstacle[3].x) / 2.0, (obstacle[0].y + obstacle[3].y) / 2.0);
            } else if (obstacleR) {
                hasRight = true;
                right1 = obstacle[1];
                right2 = Point2f((obstacle[0].x + obstacle[3].x) / 2.0, (obstacle[0].y + obstacle[3].y) / 2.0);
            }
        } else {
            obstacleL = false;
            obstacleR = false;
        }


#ifdef _DEBUG
        if (hasLeft)
            line(result, left1, left2, Scalar(0, 255, 255), 2, CV_AA);
        if (hasRight)
            line(result, right1, right2, Scalar(0, 255, 255), 2, CV_AA);
#endif

        //decide directions
        double angle = 0;
        if (hasLeft && hasRight) {

            //intersection
            double x = 0, y = 0;
            bool hasIntersection = true;
            if (fabs(left1.x - left2.x) < E && fabs(right1.x - right2.x) < E) { //两条垂直线
                hasIntersection = false;
            } else if (fabs(left1.x - left2.x) < E) { //左边界为垂直线
                double k = (right1.y - right2.y) / (right1.x - right2.x);
                x = left1.x;
                y = k * (x - right1.x) + right1.y;
            } else if (fabs(right1.x - right2.x) < E) { //右边界为垂直线
                double k = (left1.y - left2.y) / (left1.x - left2.x);
                x = right1.x;
                y = k * (x - left1.x) + left1.y;
            } else {
                double k1 = (left1.y - left2.y) / (left1.x - left2.x);
                double k2 = (right1.y - right2.y) / (right1.x - right2.x);
                if (fabs(k1 - k2) < E) {
                    //两条平行线，由于上方已过滤，不可能存在
                    hasIntersection = false;
                } else {
                    x = ((right1.y - left1.y) - (k2 * right1.x - k1 * left1.x) / (k1 - k2));
                    y = k1 * (x - left1.x) + left1.y;
                }
            }

            //求角平分线
            if (hasIntersection) {
//                double left_len = sqrt(pow(x - left2.x, 2) + pow(y - left2.y, 2));
//                double right_len = sqrt(pow(x - right2.x, 2) + pow(y - right2.y, 2));
//                double prop = left_len / right_len;
//                double x2 = (((right2.x - x) * prop + x) + left2.x) / 2;
//                double y2 = (((right2.y - y) * prop + y) + left2.y) / 2;
                Point2f bottomMid(result.cols / 2, result.rows);
                angle = atan(fabs((x - bottomMid.x) / (y - bottomMid.y))) * 180 / CV_PI;

                angle = angle > 45 ? 45 : angle;
                if (x < bottomMid.x)
                    angle = 0 - angle;

                if (obstacleR)
                    angle = 0 - fabs(angle);
                else if (obstacleL)
                    angle = fabs(angle);
#ifdef _DEBUG
                line(result, Point2f(x, y), bottomMid, Scalar(0, 255, 0), 2, CV_AA);
#endif
            } else angle = 0;
            noLinesCount = 0;
        } else if (hasLeft) {
            if (hasObstacle) {
                angle = lrightAngle;
            } else angle = rightAngle;
            noLinesCount = 0;
        } else if (hasRight) {
            if (hasObstacle)
                angle = lleftAngle;
            else angle = leftAngle;
            noLinesCount = 0;
        } else {
            noLinesCount++;
            //if(noLinesCount == 110)
            //break;
            angle = 0;
        }

#ifdef _DEBUG
        imshow("View", result);
#endif

        turnTo(angle);
        forward();
        lines.clear();
        waitKey(1);
    }

    stop();
    return 0;
}

double distance(Point2f point, double k, double b) {
    return fabs(k * point.x - point.y + b) / sqrt(k * k + 1);
}

void initialize() {
    GPIO::init();
    noLinesCount = 0;
}

void forward() {
    controlLeft(FORWARD, SET_SPEED);
    controlRight(FORWARD, SET_SPEED);
}

void stop() {
    stopLeft();
    stopRight();
}
