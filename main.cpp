#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <string>
#include <utility>
#include <thread>


using namespace std;
using namespace cv;
using namespace dnn;


struct Box_values {
    int x, y, width, height;
};


Mat model(const Mat& frame){
    auto model = readNet("../obj_det_model/frozen_inference_graph.pb","../obj_det_model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", "TensorFlow");
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);
    model.setInput(blob);
    Mat output = model.forward();
    return output;
}


vector<string> class_txt_conv() {
    vector<string> class_names;
    ifstream ifs(string("../obj_det_model/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line)){
        class_names.push_back(line);
    }
    return class_names;
}


void calc(Mat &frame, Box_values &box, int avg, int &counter, bool &checker) {
    vector<string> class_names = class_txt_conv();
    string desired_name = "person";
    Mat output = model(frame);
    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        int class_id = (int)detectionMat.at<float>(i, 1);
        float confidence = detectionMat.at<float>(i, 2);
        string current_name = class_names[class_id - 1];

        if (confidence > 0.6 && current_name == desired_name) {
            int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols - box_x);
            int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows - box_y);

            if (counter < avg) {
                cout << "\r" << (counter + 1)/(float)avg * 100 << " %" << flush;
                box.x += box_x;
                box.y += box_y;
                box.width += box_width;
                box.height += box_height;
            }
            if (counter == avg) {
                box.x /= avg;
                box.y /= avg;
                box.width /= avg;
                box.height /= avg;
            }
            if (counter > avg) {
                checker = false;
            }
        }
    }
    counter++;
}


Box_values get_box_values(const string& video, Mat frame, int avg) {
    struct Box_values box{};
        box.x = 0;
        box.y = 0;
        box.width = 0;
        box.height = 0;

    VideoCapture capture("../golf_data/" + video);
    int counter = 0;
    bool checker = true;

    while (capture.read(frame) && checker) {
        calc(frame, box, avg, counter, checker);
    }
    return box;
}


void video_display(const string& video, int avg, bool speedup, bool debug) {
    Mat frame;
    VideoCapture capture("../golf_data/" + video);
    Box_values box = get_box_values(video, frame, avg);

    int p = 0;
    while (capture.read(frame)) {
        if (frame.empty()) { break; }
        if (p % 2 or !speedup) {
            if (waitKey(10) == 't') { box.y -= 5; }
            if (waitKey(10) == 'g') { box.y += 5; }

            cv::line(frame, Point(box.x + box.width - 180, box.y), Point(box.x + box.width + 60, box.y), Scalar(0, 0, 255), 5);
            cv::line(frame, Point(box.x, box.y + 200), Point(box.x, box.y + box.height - 400), Scalar(0, 0, 255), 5);

            imshow("golf analyzer", frame);
            //imwrite(video + "_start.jpg", frame);
            if (waitKey(10) == 'q') { break; }

            if (debug) {
                if (p == 0) { waitKey(0); break; }
            }
            if (!debug) {
                if (p == 0) { waitKey(0); }
            }

        }
        p++;
    }
}

void tester() {
    for (int i = 2; i < 14; i++) {
        string video = "golf" + to_string(i) + ".MOV";
        video_display(video, 3, false, true);
        if (waitKey(10) == 'w') { break; }
    }
}

void normal(){
    string video = "golf20.mov";
    video_display(video, 3,  false, false);
}

int main(){
    //tester();
    normal();
    return 0;
}