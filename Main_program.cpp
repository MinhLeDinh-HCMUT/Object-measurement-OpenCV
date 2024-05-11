#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#define pi (2*acos(0.0));


using namespace std;
using namespace cv;

vector<Point2d> init_Points;
vector<Point2d> origin_Points;
vector<Point2d> ordered_Points;
vector<Point2d> real_Points;
double objectAngle;
double lengths[2] = { 0,0 };
Scalar color(255, 0, 0); // Define color
vector<Point2d> global(2);
int dotsize = 5; //Define dot size

vector<Point2d> getContour(Mat img, int cannyThresh[], double origin[])
{
    uint16_t kernel51[3][3] =
    {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    Mat kernel = Mat(3, 3, CV_16U, kernel51);

    Mat imggray, imgblur, imgcanny, imgdilation, imgerode;

    cvtColor(img, imggray, cv::COLOR_BGR2GRAY);
    GaussianBlur(imggray, imgblur, Size(5, 5), 1);
    Canny(imgblur, imgcanny, cannyThresh[0], cannyThresh[1]);
    dilate(imgcanny, imgdilation, kernel, Point(-1, -1), 3);
    erode(imgdilation, imgerode, kernel, Point(-1, -1), 2);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgerode, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> conPoly(contours.size());
    vector<Point2d> biggest;
    double maxArea = 5000;
    int index_max;
    size_t maxcontuorcount = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            double peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            if (area > maxArea && conPoly[i].size() == 4)
            {
                biggest = { conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3] };
                maxArea = area;
                index_max = i;
                maxcontuorcount = contours[index_max].size();
            }
        }
    }

    
    for (int i = 0; i < biggest.size(); i++) {
        circle(img, biggest[i], dotsize, color, FILLED);
    }

    vector<Moments>M(maxcontuorcount);
    if (biggest.size() > 0)
    {
        Moments M = moments(contours[index_max]);
        Point center(static_cast<int>(M.m10 / M.m00), static_cast<int>(M.m01 / M.m00));
        global[0].x = center.x;
        global[0].y = center.y;
        cout << "Centroid: " << center << endl; //c2
        circle(img, center, dotsize, color, FILLED);
        RotatedRect rotatedRect = minAreaRect(contours[index_max]);
        float  angle = rotatedRect.angle; // angle
        std::stringstream ss;
        ss << angle; // convert float to string
        cout << "Angle: " << angle << endl; //c2
    }

    namedWindow("Erosion", WINDOW_FREERATIO);
    imshow("Erosion", imgerode);

    return biggest;
}


vector<Point2d> reorderPoint(vector<Point2d> Pts, Mat img) {
    vector<Point2d> ptsWithOrder;
    vector<double>  newPoints1, newPoints2;

    for (int i = 0; i < Pts.size(); i++)
    {
        newPoints1.push_back(Pts[i].x + Pts[i].y);
        newPoints2.push_back(Pts[i].x - Pts[i].y);
    }
    ptsWithOrder.push_back(Pts[min_element(newPoints1.begin(), newPoints1.end()) - newPoints1.begin()]); //0		
    ptsWithOrder.push_back(Pts[max_element(newPoints2.begin(), newPoints2.end()) - newPoints2.begin()]); //1		
    ptsWithOrder.push_back(Pts[min_element(newPoints2.begin(), newPoints2.end()) - newPoints2.begin()]); //2		
    ptsWithOrder.push_back(Pts[max_element(newPoints1.begin(), newPoints1.end()) - newPoints1.begin()]); //3		

    for (int i = 0; i < ptsWithOrder.size(); i++)
        putText(img, to_string(i), ptsWithOrder[i], FONT_HERSHEY_PLAIN,4, color,4);
    return ptsWithOrder;
}

vector<Point2d> changeOrigin(vector<Point2d> Pts, double Origin[])
{
    for (int i = 0; i < Pts.size(); i++)
    {
        Pts[i].x -= Origin[0];
        Pts[i].y -= Origin[1];
    }
    return Pts;
}

vector<Point2d> torealWorld(vector<Point2d> Pts)
{
    vector<Point2d> Pts_out(Pts.size());
    for (int i = 0; i < Pts.size(); i++)
    {
        double rot_mat[9] = {       // Rotation matrix
        -0.0116	, -0.9997, -0.0199,
        0.9997,	  -0.0120,  0.0220,
        -0.0223,  -0.0196,	0.9996
        };
        double tran_vec[3]{         // Translation vector
        191.8741, -94.8420, 377.6861
        };
        double intr_mat[9] = {      // Intrinsic matrix
        982.5212,	0,	639.2363,
        0,	 985.1588,	351.1940,
        0,	0,	1
        };
        double pix_coo[3] = {       // Pixel coordinate
            Pts[i].x,
            Pts[i].y,
            1
        };
        double Z = 377.6861;
        Mat R(3, 3, CV_64F, rot_mat);
        Mat K(3, 3, CV_64F, intr_mat);
        Mat T(3, 1, CV_64F, tran_vec);
        Mat mp(3, 1, CV_64F, pix_coo);
        Mat mw;
        mw = R.inv() * (K.inv() * Z * mp - T);
        //cout << mw;
        Pts_out[i].x = mw.at<double>(0, 0);
        Pts_out[i].y = mw.at<double>(1, 0);
    }
    return Pts_out;
}

void calrealLength(vector<Point2d> Pts)
{
    lengths[0] = sqrt(pow((Pts[1].x - Pts[0].x), 2) + pow((Pts[1].y - Pts[0].y), 2));
    lengths[1] = sqrt(pow((Pts[2].x - Pts[0].x), 2) + pow((Pts[2].y - Pts[0].y), 2));
}


int main() {
    VideoCapture cap(2);
    Mat frame;
    int CannyThreshold[2] = {100,100};
    double origin[2] = {1142.3, 101.8};
    Point2d originPoint;
    originPoint.x=origin[0];
    originPoint.y=origin[1];
    vector<Point2d> caldistfromCent(2);
    double centlength;
    while (1)
    {
        cap >> frame;
        if (frame.empty()) break;
        circle(frame,originPoint,dotsize,Scalar(0,0,255),FILLED);
        putText(frame, "O", originPoint, FONT_HERSHEY_PLAIN,4, Scalar(0,0,255),4);
        init_Points = getContour(frame, CannyThreshold, origin);
        if (init_Points.size() > 0)
        {
            ordered_Points = reorderPoint(init_Points, frame);
            origin_Points = changeOrigin(ordered_Points, origin);
            real_Points = torealWorld(origin_Points);
            calrealLength(real_Points);
            cout << "Width: " << lengths[1] << endl << "Length: " << lengths[0] << endl; //c4
            
            global[1].x = origin[0];
            global[1].y = origin[1];
            caldistfromCent = torealWorld(global);
            centlength = sqrt(pow((caldistfromCent[1].x - caldistfromCent[0].x), 2) + pow((caldistfromCent[1].y - caldistfromCent[0].y), 2));
            cout << "Distance from origin: " << centlength << endl; //c4
            
        }
        namedWindow("Image with Point", WINDOW_FREERATIO);
        imshow("Image with Point", frame);

        if (waitKey(10) == 'q')
            break;
    }
    cap.release();
    destroyAllWindows();
}