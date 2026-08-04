#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "FastTracker.h"


#include <sys/stat.h>
#include <sys/types.h>
void make_dir_if_not_exist(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755); // rwxr-xr-x
    }
}

using namespace std;
using namespace cv;


// ================================
// Main
// ================================
int main() {
    string det_path = "./MOT17-02-DPM/det/det.txt";
    string img_dir = "./MOT17-02-DPM/img1/";
    string out_txt = "out.txt";
    string out_dir = "tracklets_draw";

    // Create output folder if it doesn't exist
    make_dir_if_not_exist(out_dir);

    ifstream infile(det_path);
    if (!infile.is_open()) {
        cerr << "Error opening detections file: " << det_path << endl;
        return -1;
    }

    ofstream outfile(out_txt);
    if (!outfile.is_open()) {
        cerr << "Error opening output file: " << out_txt << endl;
        return -1;
    }

    cout << "Tracking started..." << endl;

    FastTracker tracker(30, 30);
    string line;
    int current_frame = -1;
    vector<Object> frame_objects;

    // Function to process one frame
    auto process_frame = [&](int frame_id, const vector<Object>& objs) {
        vector<STrack> tracks = tracker.update(objs);

        // Write results to out.txt
        for (const auto& t : tracks) {
            outfile << frame_id << ","
                    << t.track_id << ","
                    << t.tlwh[0] << ","
                    << t.tlwh[1] << ","
                    << t.tlwh[2] << ","
                    << t.tlwh[3] << ","
                    << t.score << endl;
        }

        // === Draw and save frame ===
        char filename[256];
        sprintf(filename, "%s%06d.jpg", img_dir.c_str(), frame_id);
        Mat frame = imread(filename);
        if (frame.empty()) {
            cerr << "Warning: Cannot read " << filename << endl;
            return;
        }


        for (const auto& det : objs) {
            Rect det_rect(
                int(det.rect.x),
                int(det.rect.y),
                int(det.rect.width),
                int(det.rect.height)
            );
            rectangle(frame, det_rect, Scalar(0, 0, 255), 1); // red, thin box
        }


        for (const auto& t : tracks) {
            Rect track_rect(
                int(t.tlwh[0]),
                int(t.tlwh[1]),
                int(t.tlwh[2]),
                int(t.tlwh[3])
            );
            Scalar color = tracker.get_color(t.track_id);
            rectangle(frame, track_rect, color, 2);

            string label = "ID " + to_string(t.track_id);
            int baseline = 0;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            rectangle(frame, Point(track_rect.x, track_rect.y - textSize.height - 4),
                    Point(track_rect.x + textSize.width, track_rect.y), color, FILLED);
            putText(frame, label, Point(track_rect.x, track_rect.y - 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }


        int num_dets = static_cast<int>(objs.size());
        int num_tracks = static_cast<int>(tracks.size());
        string info_text = "Frame: " + to_string(frame_id) +
                        " | Detections: " + to_string(num_dets) +
                        " | Tracklets: " + to_string(num_tracks);

        putText(frame, info_text, Point(15, 30), FONT_HERSHEY_SIMPLEX,
                0.7, Scalar(255, 255, 255), 2, LINE_AA);
        putText(frame, info_text, Point(15, 30), FONT_HERSHEY_SIMPLEX,
                0.7, Scalar(0, 0, 0), 1, LINE_AA); // black shadow


        string out_path = out_dir + "/" + to_string(frame_id) + ".jpg";
        imwrite(out_path, frame);
    };

    // ================================
    // Read and process detections
    // ================================
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        vector<string> tokens;

        while (getline(ss, token, ',')) tokens.push_back(token);
        if (tokens.size() < 7) continue;

        int frame_id = stoi(tokens[0]);
        float x = stof(tokens[2]);
        float y = stof(tokens[3]);
        float w = stof(tokens[4]);
        float h = stof(tokens[5]);
        float conf = stof(tokens[6]);

        Object obj;
        obj.rect = Rect_<float>(x, y, w, h);
        obj.label = 0;
        obj.prob = conf;

        if (current_frame == -1) current_frame = frame_id;

        // When frame changes → process previous frame
        if (frame_id != current_frame) {
            process_frame(current_frame, frame_objects);
            frame_objects.clear();
            current_frame = frame_id;
        }
        frame_objects.push_back(obj);
    }

    // Process last frame
    if (!frame_objects.empty()) {
        process_frame(current_frame, frame_objects);
    }

    infile.close();
    outfile.close();

    cout << "Tracking complete." << endl;
    cout << "Results written to " << out_txt << endl;
    cout << "Annotated frames saved in " << out_dir << "/" << endl;

    return 0;
}