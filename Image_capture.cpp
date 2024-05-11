#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);  // Open default camera
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera." << endl;
        return -1;
    }

    Mat frame;
    int count = 0;
    char key = 0;
    while (true) {
        cap >> frame;  // Capture frame from camera
        imshow("Camera", frame);
        key=waitKey(1);
        if (key== 'c') {
            count+=1;
            string directory="D:///"+to_string(count)+".jpg"; // Change your calib image directory here

            imwrite(directory, frame);  // Save the image

            cout << "Image "<<count<<" saved!"<<endl;
        } 
        else if (key== 'q') break; 
    }

    cap.release();  // Release the camera
    destroyAllWindows();  // Close all OpenCV windows
    return 0;
}
