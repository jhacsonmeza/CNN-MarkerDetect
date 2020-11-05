#include <iostream>

#include <fstream>
#include <sstream>
#include <string>

#include <vector>
#include <utility> //pair, move

#include <cmath> // INFINITY

#include <torch/torch.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

using Data = std::vector<std::pair<std::string, std::vector<float>>>;


// Custom data class
class TargetData : public torch::data::datasets::Dataset<TargetData> {
public:

    TargetData(const Data& _data, bool _transform, cv::Size _sz) : data{_data}, transform{_transform}, sz{_sz} {}

    torch::data::Example<> get(size_t idx)
    {
        cv::Mat im = cv::imread(data[idx].first);
        std::vector<float> target(data[idx].second);

        // Resize image and target
        resize_im_bbox(im, target, sz);

        // Data augmentation
        if (transform)
            transformer(im, target, 0.5, sz);

        // Image from [0,255] to [0,1]
        im.convertTo(im, CV_32FC3, 1.f/255.f);

        // Target from [0,w] and [0,h] to [0,1]
        target[0] = target[0]/sz.width;
        target[1] = target[1]/sz.height;
        target[2] = target[2]/sz.width;
        target[3] = target[3]/sz.height;

        // Convert image to torch tensor
        auto tdata = torch::from_blob(im.data, {im.rows, im.cols, 3}, torch::kFloat32).clone();
        tdata = tdata.permute({2,0,1});

        // Convert label to torch tensor
        auto tlabel = torch::from_blob(target.data(), {static_cast<int>(target.size())}, torch::kFloat32).clone();

        return {tdata, tlabel};
    }

    torch::optional<size_t> size() const
    {
        return data.size();
    }

private:
    Data data;
    bool transform;
    cv::Size sz;
};

int main()
{
    // Defining device: CPU or GPU
    torch::DeviceType device;
    if (torch::cuda::cudnn_is_available())
    {
        std::cout << "CUDA is available. Training on GPU..." << std::endl;
        device = torch::kCUDA;
        // auto device = torch::Device(torch::kCUDA, 0);
    }
    else
    {
        std::cout << "Training on CPU..." << std::endl;
        device = torch::kCPU;
    }


    // Load csv data
    std::ifstream file("data_bbox.csv"); // Stream to the input csv file

    // Read first line and ignore (title)
    std::string line;
    std::getline(file, line); //file >> line; split in white spaces
    
    Data data;
    while (std::getline(file, line)) // Load data
    {
        std::stringstream ss(line);

        std::string filename;
        std::getline(ss,filename,',');
        
        std::vector<float> labels(4);
        for (int i = 0; i < 4; i++)
        {
            // Save xc, yc, width, height values
            std::string label;
            std::getline(ss,label,',');
            labels[i] = std::stof(label);
        }
        
        data.push_back({filename, labels});
    }

    // Split data into training and test
    int n = data.size();
    int len_train = int(0.9*n);
    int len_test{n-len_train};

    Data data_training(data.begin(), data.begin()+len_train);
    Data data_test(data.end()-len_test, data.end());
    
    // Split training data into training and validation
    n = len_train;
    len_train = int(0.9*n);
    int len_val{n-len_train};

    Data data_train(data_training.begin(), data_training.begin()+len_train);
    Data data_val(data_training.end()-len_val, data_training.end());

    std::cout << "Train data length: " << len_train << std::endl;
    std::cout << "Validation data length: " << len_val << std::endl;
    std::cout << "Test data length: " << len_test << std::endl << std::endl;


    // Instantiating dataset class
    auto train_ds = TargetData(data_train,true,cv::Size(224,224)).map(torch::data::transforms::Stack<>());
    auto val_ds = TargetData(data_val,false,cv::Size(224,224)).map(torch::data::transforms::Stack<>());

    // Creating data loaders
    auto train_dl = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_ds),torch::data::DataLoaderOptions(64).workers(4));
    auto val_dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(val_ds),torch::data::DataLoaderOptions(32).workers(4));


    // Create the model
    auto model = vision::models::ResNet18();
    torch::load(model,"resnet18.pt"); // Load ImageNet pretrained weights
    model->fc = model->replace_module("fc", torch::nn::Linear(512, 4));
    for (auto& v: model->parameters())
		v.set_requires_grad(true);
    torch::load(model, "bbox++.pt");
    model->to(device);

    // Defining the loss function
    torch::nn::SmoothL1Loss loss_func(torch::nn::SmoothL1LossOptions().reduction(torch::kSum));

    // Defining the optimizer
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(3e-4));

    // Learning rate schedule
    auto& options = static_cast<torch::optim::AdamOptions&>(opt.param_groups()[0].options());
    double factor = 0.5;
    int patience = 20;


    //--------------------------------- Trining loop
    int n_epochs = 200, count_lr = 0;
    float val_loss_min = INFINITY;
    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        // Get value of the current learning rate
        double current_lr = options.lr();

        // keep track of training and validation loss
        float train_loss = 0.0, val_loss = 0.0;
        float train_metric = 0.0, val_metric = 0.0;

        // Train the model
        model->train();
        for (auto& batch : *train_dl)
        {
            auto xb = batch.data.to(device); //(n,3,224,224)
            auto yb = batch.target.to(device); //(n,4)

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto output = torch::sigmoid(model->forward(xb)); //(n,4)
            // Calculate the batch loss
            auto loss = loss_func(output, yb); //[(n,4),(n,4)] -> integer

            // Clear the gradients of all optimized variables
            opt.zero_grad();
            // Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward();
            // Perform a single optimization step (parameter update)
            opt.step();

            // Update train loss
            train_loss += loss.item<float>();

            // Update metric (for acc)
            train_metric += iou_bbox(reinterpret_cast<float*>(output.cpu().data_ptr()), reinterpret_cast<float*>(yb.cpu().data_ptr()), xb.sizes().data());
        }

        // Validate the model
        model->eval();
        torch::NoGradGuard no_grad;
        for (auto& batch : *val_dl)
        {
            auto xb = batch.data.to(device); //(n,3,224,224)
            auto yb = batch.target.to(device); //(n,4)

            // Forward pass: compute predicted outputs by passing inputs to the model
            auto output = torch::sigmoid(model->forward(xb)); //(n,4)
            // Calculate the batch loss
            auto loss = loss_func(output, yb); //[(n,4),(n,4)] -> integer

            // Update validation loss
            val_loss += loss.item<float>();

            // Update metric (for acc)
            val_metric += iou_bbox(reinterpret_cast<float*>(output.cpu().data_ptr()), reinterpret_cast<float*>(yb.cpu().data_ptr()), xb.sizes().data());
        }

        // Calculate average losses of the epoch
        train_loss /= len_train;
        val_loss /= len_val;

        train_metric /= len_train;
        val_metric /= len_val;


        // Store best model - learning rate schedule
        if (val_loss < val_loss_min)
        {
            std::cout << "Validation loss decreased (" << val_loss_min << " --> " << val_loss << "). Saving model..." << std::endl;
            torch::save(model, "weights.pt");

            val_loss_min = val_loss;
            count_lr = 0; // Reset counter
        }
        else if (++count_lr == patience)
        {
            std::cout << "Reducing learning rate to " << current_lr*factor <<  std::endl;
            options.lr(current_lr*factor);

            std::cout << "Loading best model weights" <<  std::endl;
            torch::load(model, "weights.pt");

            count_lr = 0; // Reset counter
        }


        std::cout << "Epoch " << epoch+1 << "/" << n_epochs << ", "
        << "lr = " << current_lr << ", "
        << "train loss: " << train_loss << ", val loss: " << val_loss
        << ", train acc: " << 100*train_metric << "% , val acc: " << 100*val_metric << "%" << std::endl;

        std::cout << "----------" << std::endl;
    }
    
    return 0;
}
