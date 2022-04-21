#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <string>
#include <utility>
#include <thread>
#include <typeinfo>
#include <fstream>


using namespace std;
using namespace cv;
using namespace dnn;

string DIR_PATH = "/Users/markusjonek/Documents/golf-analys";

struct Box_values {
    int x, y, width, height;
};


vector<string> class_txt_converter() {
    vector<string> class_names;
    ifstream ifs(string(DIR_PATH + "/data/object_detection_classes_coco.txt").c_str());
    string line;
    while (getline(ifs, line)){
        class_names.push_back(line);
    }
    return class_names;
}


Mat object_detection_model(const Mat& frame){
    auto model = readNet(DIR_PATH + "/data/frozen_inference_graph.pb", DIR_PATH + "/data/ssd_mobilenet_v2_coco_2018_03_29.pbtxt", "TensorFlow");
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);
    model.setInput(blob);
    Mat output = model.forward();
    return output;
}


void calculate_box_values(Mat &frame, Box_values &box, int avg, int &counter, bool &checker) {
    vector<string> class_names = class_txt_converter();
    string desired_name = "person";
    Mat model_output = object_detection_model(frame);
    Mat detectionMat(model_output.size[2], model_output.size[3], CV_32F, model_output.ptr<float>());

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
                //cout << "\r" << (counter + 1)/(float)avg * 100 << " %" << flush;
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


Box_values get_avg_box_values(const string& video_path, Mat frame, int avg) {
    struct Box_values box{};
        box.x = 0;
        box.y = 0;
        box.width = 0;
        box.height = 0;

    //VideoCapture capture("../golf_data/" + video);
    VideoCapture capture(video_path);
    int counter = 0;
    bool checker = true;

    while (capture.read(frame) && checker) {
        calculate_box_values(frame, box, avg, counter, checker);
    }
    return box;
}


void video_writer(const string& video_path, bool speedup) {
    Mat frame;
    //VideoCapture capture("../golf_data/" + video);
    VideoCapture capture(video_path);
    Box_values box = get_avg_box_values(video_path, frame, 5);

    VideoWriter writer;
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');  // select desired codec (must be available at runtime)
    double fps = 25.0;                          // framerate of the created video stream
    string filename = DIR_PATH + "/analyzed_swing.mp4";             // name of the output video file
    //bool isColor = (frame.type() == CV_8UC3);
    capture >> frame;
    writer.open(filename, codec, fps, frame.size(), true);

    int p = 0;
    while (capture.read(frame)) {
        if (frame.empty()) { break; }
        if (p % 2 or !speedup) {
            if (waitKey(10) == 't') { box.y -= 5; }
            if (waitKey(10) == 'g') { box.y += 5; }

            cv::line(frame, Point(box.x + box.width - 180, box.y), Point(box.x + box.width + 60, box.y), Scalar(0, 0, 255), 5);
            cv::line(frame, Point(box.x, box.y + 200), Point(box.x, box.y + box.height - 400), Scalar(0, 0, 255), 5);
            writer.write(frame);
            //imshow("golf analyzer", frame);
            //if (waitKey(1) == 'q') { break; }
            //if (p == 0) { waitKey(0); } // vänta på mellanslag för att börja

        }
        p++;
    }
}



int main(int argc, char *argv[]){
    if ('/' == argv[1][0]) {
        video_writer(argv[1],  false);
    } 
    else {
        string vid = argv[1];
        video_writer(DIR_PATH + "/golf_videos/" + vid,  false);
    }
    system(("open " + DIR_PATH + "/analyzed_swing.mp4").c_str());
    return 0;
}




