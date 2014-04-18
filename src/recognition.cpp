#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main() {
    
    string fn_haar = "haarcascade_frontalface_alt.xml";
    string fn_csv = "faces1.csv";
    vector<Mat> images;
    vector<int> labels;
    read_csv(fn_csv, images, labels);
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    VideoCapture cap(2);
    Mat frame;

    Mat check;
    //check = imread("Database/MAYANK/Sat Nov  2 14:51:09 2013.jpg");
    for(;;) {
        cap >> frame;
        Mat original = frame.clone();
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        Mat face_resized;
        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            int prediction = model->predict(face_resized);
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            string box_text = format("Prediction = %d", prediction);
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
       
		imshow("face_recognizer", original);
		 
        char key = (char) waitKey(20);
		if (key == 27){
			break;
		}
    }
    return 0;
 }
