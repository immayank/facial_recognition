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
    vector<Mat> images;
    vector<int> labels;
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    VideoCapture cap(0);
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
        Mat face;
        Mat face_resized;
        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            face = gray(face_i);
            cv::resize(face, face_resized, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
        }
        if (face_resized.cols > 0){
			imshow("face_recognizer", face_resized);
		}
		else 
		cout<<"No face Detected \n";
        char key = (char) waitKey(20);
        if(key == 32){
		string filename;
		cout << "Training..  Enter Name in Caps: ";
		string input;
		string name;
		cin >> name;
		input = name;
        cout<< name << " entered into data \n";
        filename.append("Database/");
		filename.append(name);
		char *a=new char[filename.size()+1];
		a[filename.size()]=0;
		memcpy(a,filename.c_str(),filename.size());
		mkdir(a,0777);
		time_t now = time(0);
		char* dt = ctime(&now);
		filename.append("/");
		filename.append(dt);
		filename.erase(filename.end()-1,filename.end());
		filename.append(".jpg");
        imwrite(filename,face_resized);
        cout << "Data stored in " << filename <<"\n"<<endl;	
        filename.append(";");
        filename.append(input);
        ofstream outfile;
		outfile.open("faces.csv", std::ios_base::app);
		outfile << filename<<"\n";
        
		}
		else if (key == 27){
		break;
		}
    }
    return 0;
 }
