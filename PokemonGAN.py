import os
import csv
import time
import math
import torch
import imageio # ?
import numpy as np
import pandas as pd
from torch import nn
from os import listdir
from skimage import io # ?
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from os.path import isfile, join, isdir
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted, ns, natsort_keygen
from torchvision.utils import make_grid

import fid

torch.manual_seed(36)

common_image_extensions = [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi", ".png", ".gif", ".webp", ".tiff", ".tif", ".psd", ".raw", ".arw", ".cr2", ".nrw", ".k25", ".bmp", ".dib", ".heif", ".heic", ".ind", ".indd", ".indt", ".jp2", ".j2k", ".jpf", ".jpx", ".jpm", ".mj2", ".svg", ".svgz", ".ai", ".eps"]
# Helper class for handling file and directory names #
class FFile:
    def __init__(self, full, file_or_folder):
        # Call to get the type
        self.file_or_folder = file_or_folder
        # Full path and file
        self.full = full
        # Path without file
        self.path = os.path.dirname(full)
        # File name + extension
        self.full_file_name = os.path.basename(full)
        if self.file_or_folder == "File" or self.file_or_folder == "file":
            self.file_name, self.file_extension = os.path.splitext(self.full_file_name)
        else:
            # File name without extension
            self.file_name = self.full_file_name
            # Extension
            self.file_extension = ''


# Prints checks for cuda and cudnn and sets up the device #
def soft_control():
    # This flag controls whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores, #
    # available on new NVIDIA GPUs since Ampere, internally to compute matmul (matrix multiplies and batched matrix multiplies) and convolutions. #
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True. #
    torch.backends.cuda.matmul.allow_tf32 = True
    print("TensorFloat32 tensor cores allowed on cuda.")
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True. #
    torch.backends.cudnn.allow_tf32 = True
    print("TensorFloat32 tensor cores allowed on cuDNN.")

    # Check whether cuda is available and its version #
    print("Cuda availability: " + str(torch.cuda.is_available()))
    print("Cuda version: " + str(torch.version.cuda))

    # Check whether cudnn is available and its version #
    print("CuDNN availability: " + str(torch.backends.cudnn.enabled))
    print("CuDNN version: " + str(torch.backends.cudnn.version()))

    # Set the main device for calculations #
    if torch.cuda.is_available() is True:
        device = 'cuda'
    else:
        device = 'cpu'

    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return device


# Helper function for checking whether a tensor is on cuda or not #
def cuda_check(tensor_check):
    if tensor_check.is_cuda:
        print("Variable is on cuda.")
    else:
        print("Variable is not on cuda.")


# Helper function for viewing loaded images in a grid #
def show_tensor_images(image_tensor, num_images=25, size=(3, 96, 96), upcolor=True):
    # Highlight colours better #
    if upcolor is True:
        image_tensor = (image_tensor + 1) / 2

    # Convert to plt convention and make a grid of images #
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


# Main training function #
def training(paths, parameters, misc, device):
    n_classes = 1
    class_names = ["Pokemon"]


    # Start measuring time #
    before = time.time()

    # Compose transforms for loading dataset #
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    ])

    # Creating and loading dataset #
    dataset = PokemonDataset(paths["path_csv_database"], paths["path_resized_original_images"], transform=transform, remove_alpha=False)
    dataloader = DataLoader(dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=4)

    # Initializing generator and critic #
    generator = Generator(kernels=parameters["generator_kernels"], strides=parameters["generator_strides"], input_dim=parameters["z_dim_generator"]).to(device)
    critic = Critic(kernels=parameters["critic_kernels"], strides=parameters["critic_strides"], image_channels=parameters["image_channels"]).to(device) 

    # Initializing optimization method #
    if parameters["optim"] == 'Adam':
        generator_opt = torch.optim.Adam(generator.parameters(), lr=parameters["lr"], betas=(parameters["beta_1"], parameters["beta_2"]))
        critic_opt = torch.optim.Adam(critic.parameters(), lr=parameters["lr"], betas=(parameters["beta_1"], parameters["beta_2"]))
    else:
        if parameters["optim"] == 'RMSprop':
            generator_opt = torch.optim.RMSprop(generator.parameters(), lr=parameters["lr"], momentum=parameters["beta_1"])
            critic_opt = torch.optim.RMSprop(critic.parameters(), lr=parameters["lr"], momentum=parameters["beta_1"])
    
    # Misc variable initialization #
    losses_generator = []
    losses_critic = []
    losses_steps = 0
    epoch_steps = [0]

    # Main loop over epochs #
    for epoch in range(parameters["n_epochs"]):
        # Batch loop #
        for real, labels in dataloader:
            cur_batch_size = len(real)
            real = real.to(device)

            mean_iteration_critic_loss = 0
            for _ in range(parameters["critic_repeats"]):
                # Update critic #
                critic_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, parameters["z_dim_generator"], device=device)

                fake = generator(fake_noise)

                critic_fake_pred = critic(fake.detach())
                critic_real_pred = critic(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                critic_loss = get_critic_loss(critic_fake_pred, critic_real_pred, gp, parameters["c_lambda"])
                mean_iteration_critic_loss = mean_iteration_critic_loss + critic_loss

                # Update gradients #
                critic_loss.backward(retain_graph=True)
                # Update the weights #
                critic_opt.step()

            # Update generator #
            generator_opt.zero_grad()

            noise_2 = get_noise(cur_batch_size, parameters["z_dim_generator"], device=device)
            fake_2 = generator(noise_2)
            critic_fake_pred = critic(fake_2)
        
            generator_loss = get_generator_loss(critic_fake_pred)
            # Update gradients #
            generator_loss.backward()

            # Update the weights #
            generator_opt.step()

            # Keep track of the average loss #
            critic_loss_calc = mean_iteration_critic_loss / parameters["critic_repeats"]
            losses_critic.append(critic_loss_calc.item())
            losses_generator.append(generator_loss.item())
            losses_steps = losses_steps + 1

        # Save critic and generator states every {save_step} epochs #
        if (epoch+1) % misc["save_step"] == 0:
            save_nn_state(generator, critic, paths["path_training_network_critics"], paths["path_training_network_generators"], misc["dt_string"], misc["learning_set_name"], epoch)

        # Console info and axis update #
        print("Epoch: " + str(epoch+1))
        epoch_steps.append(losses_steps)

    # Save parameters and info to .txt file #
    after = time.time()
    time_spent = (after - before) / 60
    save_info_to_file(paths["path_training_info"], misc, parameters, time_spent)

    # Save image plot of losses #
    create_matplot(losses_steps, losses_generator, losses_critic, epoch_steps, paths["path_training_info"])

    create_results(parameters, misc, n_classes, device, class_names, paths, parameters["z_dim_generator"], dataset)


def create_results(parameters, misc, n_classes, device, class_names, paths, generator_input_dim, dataset):
    training_progress_full_images = []
    training_progress_images = []
    for _ in range(n_classes):
        training_progress_images.append([])

    average_fid = []
    fid_values = []
    for i in range(parameters["n_epochs"]):
        fid_values.append([])
        for _ in range(n_classes):
            fid_values[i].append([])

    iteration = 0
    for epoch in range(misc["save_step"], parameters["n_epochs"] + misc["save_step"], misc["save_step"]):
        path = os.path.join(paths["path_training_network_generators"], str('generator_' + misc["dt_string"] + '_' + misc["learning_set_name"] + '_' + str(epoch) + '.pt'))
        model = Generator(kernels=parameters["generator_kernels"], strides=parameters["generator_strides"], input_dim=generator_input_dim).to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        # Generate training progress images for each class per epoch #
        training_progress_images, training_progress_full_images = collect_training_images_for_gif(n_classes, parameters["z_dim_generator"], model, training_progress_images, training_progress_full_images, device)

        # Calculate FID and save critic and generator states every (save_step) epochs #
        os.mkdir(os.path.join(paths["path_training_images"], str(epoch)))
        os.mkdir(os.path.join(paths["path_training_images_grids"], str(epoch)))
        generate_save_gen_samples(os.path.join(paths["path_training_images"], str(epoch)), os.path.join(paths["path_training_images_grids"], str(epoch)), dataset, misc["generation_nr"], n_classes, parameters["z_dim_generator"], model, device)
        average_fid_epoch, fid_values_epoch = calculate_fids(class_names, paths["path_resized_original_images"], os.path.join(paths["path_training_images"], str(epoch)), misc["generation_nr"], 1)
        average_fid.append(average_fid_epoch)
        #print(fid_values_epoch)
        for i in range(n_classes):
            fid_values[iteration][i] = fid_values_epoch[i]
            print(fid_values[iteration][i])

        # Generate and save a handful of image samples #
        generate_save_gen_samples(paths["path_training_images"], paths["path_training_images_grids"], dataset, misc["generation_nr"], n_classes, parameters["z_dim_generator"], model, device)

        iteration = iteration + 1

    file = open(os.path.join(paths["path_training_info"], "Info.txt"), "a", encoding="utf-8")
    leaps = math.ceil(parameters["n_epochs"] / misc["save_step"])
    leap_length = parameters["n_epochs"] / leaps
    i = 0
    for saved_step in range(leaps):
        j = 0
        file.write("\n")
        file.write("Epoch: " + str(int((i+1) * leap_length)) + '\n')
        file.write("Average FID: " + str(average_fid[i]) + '\n')
        for class_name in class_names:
            file.write("FID Value " + class_name + ": " + str(fid_values[i][j]) + '\n')
            j = j + 1

        i = i + 1

    file.close()

    # Save gif images from generator #
    save_gif_images(n_classes, class_names, paths["path_training_gifs"], training_progress_images, training_progress_full_images)

    # Show results of all classes in a grid #
    image = training_progress_full_images[-1]
    image.save(os.path.join(paths["path_training_images_grids"], "AllClassesResultGrid.png"))


# Calculate Frechet Inception Distances of all class names #
def calculate_fids(class_names, path_resized_original_images, path_training_images, generation_nr, workers_n):
    fid_values = []
    # Comparing resized original images with generated images #
    for class_name in class_names:
        paths = []
        paths.append(os.path.join(path_resized_original_images, class_name))
        paths = [path_resized_original_images]
        paths.append(os.path.join(path_training_images, class_name))
        fid_values.append(fid.calculate_fid_given_paths(paths, generation_nr, "cuda", 2048, workers_n))

    average_fid = sum(fid_values) / len(fid_values)

    return (average_fid, fid_values)


# Generate and save generator samples separately and in a grid #
def generate_save_gen_samples(path_training_images, path_training_images_grids, dataset, generation_nr, n_classes, z_dim_generator, generator, device):
    class_index = 0
    class_samples = []
    class_names = dataset.get_class_names()
    for _ in range(len(class_names)):
        class_samples.append([])

    # Loop over all classes #
    for class_name in class_names:
        # Create subdirectories #
        if os.path.isdir(os.path.join(path_training_images, class_name)) is False:
            os.mkdir(os.path.join(path_training_images, class_name))

        path_training_images_class = os.path.join(path_training_images, class_name)
        tensor_samples = []

        # Generate int(generation_nr) samples of a specific class and save them to subdir #
        # Crop_circle only allows Image data type so I'm converting to Image and afterwards back to Tensor #
        for i in range(generation_nr):
            tensor_sample = sample_generator(class_index, n_classes, z_dim_generator, generator, device=device)
            image_sample = tensor_to_Image(tensor_sample, normalize=True)
            #image_sample = crop_circle(image=image_sample)
            tensor_sample = transforms.ToTensor()(image_sample).unsqueeze_(0)
            tensor_samples.append(tensor_sample)
            image_name = "gen_" + class_name + "_" + str(i)
            image_sample.save(os.path.join(path_training_images_class, image_name + ".png"))

        # Make grid from the generated samples as one image and save it #
        images_combined = make_tensor_grid(tensor_samples, 8)
        image = tensor_to_Image(images_combined, normalize=True)
        image.save(os.path.join(path_training_images_grids, class_name + "ResultGrid.png"))
        class_index = class_index + 1


# Save training gif images #
def save_gif_images(n_classes, class_names, path_training_gifs, training_progress_images, training_progress_full_images):
    # Loop over all class names #
    j = 0
    for class_name in class_names:
        # Create subdir #
        if os.path.isdir(os.path.join(path_training_gifs, class_name)) is False:
            os.mkdir(os.path.join(path_training_gifs, class_name))

        # Save gifs of separate classes training #
        full_file_name = os.path.join(path_training_gifs, class_name, (class_name + "_training.gif"))
        imageio.mimwrite(full_file_name, training_progress_images[j], fps=2)
        j = j + 1

    for i in range(len(training_progress_full_images)):
        training_progress_full_images[i] = tensor_to_Image(training_progress_full_images[i], normalize=True)

    # Save gif of all classes training in a grid #
    full_file_name = os.path.join(path_training_gifs, 'training_CGAN_AllClasses.gif')
    imageio.mimwrite(full_file_name, training_progress_full_images, fps=2)


# Generate images for training gif #
def collect_training_images_for_gif(n_classes, z_dim_generator, generator, training_progress_images, training_progress_full_images, device):
    desired_images = []
    desired_grid_images = []
    training_full_list = []

    # Loop over all classes to sample generator #
    for desired_class in range(n_classes):
        desired_images.append(sample_generator(desired_class, n_classes, z_dim_generator, generator, device=device))
        desired_image = tensor_to_Image(desired_images[desired_class], normalize=True)
        #desired_image = crop_circle(image=desired_image)
        tensor = transforms.ToTensor()(desired_image).unsqueeze_(0)
        training_full_list.append(tensor)
        desired_grid_images.append(desired_image)
        training_progress_images[desired_class].append(desired_grid_images[desired_class])

    # Make a one-image grid #
    training_full_grid = make_tensor_grid(training_full_list, 8)
    training_progress_full_images.append(training_full_grid)

    return (training_progress_images, training_progress_full_images)


# Makes a grid of tensor images as a one image #
def make_tensor_grid(tensor_list, column_n):
    length = len(tensor_list)
    rows_n = math.ceil(length / column_n)
    rows = []

    # Adds images horizontally to create a row #
    for row_n in range(rows_n):
        start = 0 + row_n * column_n
        stop = column_n + row_n * column_n

        # Fill tensor list with blanks so that rows can be stacked without hiccups
        if stop > length:
            for additional in range(stop-length):
                tensor_list.append(torch.zeros_like(tensor_list[0]))

        rows.append(torch.cat((tensor_list[start:stop]), 3))

    # Adds rows vertically to create columns #
    stacked = torch.cat((rows[:]), 2)

    return stacked


# Create an image by nicely asking a friendly generator #
def sample_generator(desired_class, n_classes, z_dim_generator, generator, device):
    label = torch.nn.functional.one_hot(torch.Tensor([desired_class]).long().cuda(), n_classes).float()
    noise = get_noise(1, z_dim_generator, device=device)
    noise_and_labels = combine_vectors(noise, label)

    return generator(noise)


# Create and save matplot visualizing generator and critic losses #
def create_matplot(losses_steps, losses_generator, losses_critic, epoch_steps, path_training_info):
    # Clear figure, set labels, legend and axes #
    plt.figure(0)
    plt.clf()
    plt.plot(range(losses_steps), losses_generator, label = 'Generator Losses')
    plt.plot(range(losses_steps), losses_critic, label = 'Critic Losses')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    axis = plt.axis()

    # Create green vertical lines dividing the steps into ~10 visual sections #
    v_distance = math.floor(len(epoch_steps) / 10)
    for step in range(10):
        plt.vlines(epoch_steps[step*v_distance], axis[2], axis[3], color='g')

    # Save #
    full_file_path = os.path.join(path_training_info, 'losses.png')
    plt.savefig(full_file_path, bbox_inches='tight')


# Save critic and generator states #
def save_nn_state(generator, critic, path_critics, path_generators, dt_string, learning_set_name, epoch):
    torch.save(critic.state_dict(), os.path.join(path_critics, str('critic_' + dt_string + '_' + learning_set_name + '_' + str(epoch+1) + '.pt')))
    torch.save(generator.state_dict(), os.path.join(path_generators, str('generator_' + dt_string + '_' + learning_set_name + '_' + str(epoch+1) + '.pt')))


# Save GAN parameters to txt file #
def save_info_to_file(path_training_info, misc, parameters, time_spent):
    file = open(os.path.join(path_training_info, "Info.txt"), "w", encoding="utf-8")
    file.write(misc["dt_string"] + '\n')
    file.write(misc["learning_set_name"] + '\n')
    file.write("Time spent: " + str(time_spent) + '\n')
    file.write("Epochs: " + str(parameters["n_epochs"]) + '\n')
    file.write("Batch size: " + str(parameters["batch_size"]) + '\n')
    file.write("Learning Rate: " + str(parameters["lr"]) + '\n')
    file.write("Beta 1: " + str(parameters["beta_1"]) + '\n')
    file.write("Beta 2: " + str(parameters["beta_2"]) + '\n')
    file.write("c_lambda: " + str(parameters["c_lambda"]) + '\n')
    file.write("Critic Repeats: " + str(parameters["critic_repeats"]) + '\n')
    file.write("Optimization: " + str(parameters["optim"]) + '\n')
    #file.write("Average FID: " + str(average_fid) + '\n')

    #leaps = math.ceil(parameters["n_epochs"] / misc["save_step"])
    #leap_length = parameters["n_epochs"] / leaps
    #i = 0
    #for saved_step in range(leaps):
    #    j = 0
    #    file.write("\n")
    #    file.write("Epoch: " + str(int((i+1) * leap_length)) + '\n')
    #    file.write("Average FID: " + str(average_fid[i]) + '\n')
    #    for class_name in class_names:
    #        file.write("FID Value " + class_name + ": " + str(fid_values[int((i+1) * leap_length - 1)][j]) + '\n')
    #        j = j + 1

    #    i = i + 1

    file.close()


# Rename images in a directory to a directory's name + index #
def rename_images_in_folder(path):
    # Helper class #
    main_folder_FF = FFile(path, "folder")

    # Find all files in a directory and loop over them #
    file_names = [f for f in listdir(path) if isfile(join(path, f))]
    i = 0
    for full_file_name in file_names:
        file_FF = FFile(full_file_name, "file")
        if file_FF.file_extension in common_image_extensions:
            if os.path.isfile(os.path.join(path, str(main_folder_FF.file_name + "_" + str(i) + file_FF.file_extension))) is False:
                os.rename(os.path.join(path, file_FF.full_file_name), os.path.join(path, str(main_folder_FF.file_name + "_" + str(i) + file_FF.file_extension)))
                i = i + 1


# Dataset class #
class PolandballDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, remove_alpha=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.y_labels = []
        self.class_names = []
        self.class_num = {}

        # Loop over csv names to load images #
        for index in range(len(self.annotations)):
            img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].split("_", 1)[0], self.annotations.iloc[index, 0])
            image_path_FF = FFile(img_path, "file")

            # # Assessing class names
            # if image_path_FF.file_name.split("_", 1)[0] not in self.class_names:
            #     self.class_names.append(image_path_FF.file_name.split("_", 1)[0])
            #     self.class_num[str(image_path_FF.file_name.split("_", 1)[0])] = 1
            # else:
            #     self.class_num[str(image_path_FF.file_name.split("_", 1)[0])] = self.class_num[str(image_path_FF.file_name.split("_", 1)[0])] + 1

            image = io.imread(img_path)

            # Remove alpha channel if you so wish
            if remove_alpha is True:
                image = image[:,:,:3]

            self.y_labels.append(torch.tensor(self.annotations.iloc[index, 1]))

            if self.transform:
                image = self.transform(image)

            self.images.append(image)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return (self.images[index], self.y_labels[index])

    def get_file_name(self, index):
        return self.annotations.iloc[index, 0]

    def get_n_classes(self):
        return int(self.annotations.iloc[-1, 1]) + 1

    # def get_class_names(self):
    #     return self.class_names

    def get_class_num(self, name):
        return self.class_num[name]


class PokemonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, remove_alpha=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.class_names = ["Pokemon"]

        # Loop over csv names to load images #
        for index in range(len(self.annotations)):
            img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
            image = io.imread(img_path)

            # Remove alpha channel if you so wish
            if remove_alpha is True:
                image = image[:,:,:3]

            if self.transform:
                image = self.transform(image)

            self.images.append(image)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return (self.images[index], 0)

    def get_file_name(self, index):
        return self.annotations.iloc[index, 0]

    def get_class_names(self):
        return self.class_names


# Creates a .csv file containing info on files in a directory #
def create_database_csv(csv_file_path, database_path, file_name):
    # Open file, loop over all paths to write file names to the csv file #
    with open(os.path.join(csv_file_path, file_name), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        full_file_names = [f for f in listdir(database_path) if isfile(join(database_path, f))]
        for full_file_name in full_file_names:
            full_file_name_FF = FFile(full_file_name, "file")
            if full_file_name_FF.file_extension in common_image_extensions:
                writer.writerow([full_file_name])


# Helper function for calculaing the dimensions of generator output #
def calculate_gen_output(value, stride, padding, kernel, dilation=1):
    for i in range(len(stride)):
        value = (value - 1) * stride[i] - 2 * padding[i] + dilation * (kernel[i] - 1) + 1

    return value


# Helper function for calculaing the dimensions of critic output #
def calculate_crit_output(value, stride, kernel):
    for i in range(len(stride)):
        value = math.floor((value - kernel[i]) / (stride[i]) + 1)

    return value


# Convert tensor to Image (uint8)
def tensor_to_Image(tensor, normalize=False):
    #tensor.requires_grad = False
    tensor = tensor.detach().cpu().numpy()
    tensor = np.squeeze(tensor)
    tensor = tensor.transpose(1, 2, 0)
    if normalize is True:
        tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))

    tensor = tensor * 255
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)

    return image


def prepare_database(path_original_images, path_resized_original_images, path_augmented_images, image_dimensions):
    original_ball_folder_FF = FFile(path_original_images, "folder")
    create_database_csv(original_ball_folder_FF.full, original_ball_folder_FF.full, str(original_ball_folder_FF.file_name + "Database.csv"))
    path_csv = os.path.join(original_ball_folder_FF.full, str(original_ball_folder_FF.file_name + "Database.csv"))
    if (os.path.isdir(path_resized_original_images)) is False:
        os.mkdir(path_resized_original_images)
    if (os.path.isdir(path_augmented_images)) is False:
        os.mkdir(path_augmented_images)

    read = pd.read_csv(path_csv)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_dimensions, image_dimensions))
        ])

    dataset = PokemonDataset(path_csv, path_original_images, transform=transform)
    i = 0
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4)
    for real, _ in dataloader:
        image = tensor_to_Image(real)
        image.save(os.path.join(path_resized_original_images, dataset.get_file_name(i)))
        i = i + 1

    # augmentacja tutaj powinna byc




# Move all files from a directory of directories to another directory #
def move_files_from_folders(folder_main, folder_destination):
    # Loop over folders in directory #
    ball_folders = [f.path for f in os.scandir(folder_main) if f.is_dir()]
    for ball_folder in ball_folders:
        # Loop over files in directories #
        file_names = [f for f in listdir(ball_folder) if isfile(join(ball_folder, f))]
        for file_name in file_names:
            # Move files to destination, "unpacking" them #
            file_name_source_FF = FFile(os.path.join(ball_folder, file_name), "file")
            file_name_destination_FF = FFile(os.path.join(folder_destination, file_name), "file")
            os.replace(file_name_source_FF.full, file_name_destination_FF.full)


# Combines two vectors #
def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)

    return combined


# Calculates the critic loss #
def get_critic_loss(critic_fake_pred, critic_real_pred, gp, c_lambda):
    return torch.mean(critic_fake_pred) - torch.mean(critic_real_pred) + c_lambda * gp


# Calculates the gradient penalty of a gradient #
# Given a batch of image gradients, you calculate the magnitude of each image's gradient #
# and penalize the mean quadratic distance of each magnitude to 1. #
def gradient_penalty(gradient):
    # Flatten the gradients so that each row captures one image #
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row #
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1 #
    penalty = torch.mean((gradient_norm - 1)**2)

    return penalty


# Calculates the generator loss #
def get_generator_loss(critic_fake_pred):
    return -1. * torch.mean(critic_fake_pred)


# Return the gradient of the critic's scores with respect to mixes of real and fake images #
def get_gradient(crit, real, fake, epsilon):
    # Mix the images together #
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images # 
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images #
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )

    return gradient[0]


# Function for getting the size of the conditional input dimensions #
def get_input_dimensions(generator_input_dim, critic_image_channels, n_classes):
    generator_input_dim_2 = generator_input_dim + n_classes
    discriminator_im_chan = critic_image_channels + n_classes

    return generator_input_dim_2, discriminator_im_chan


# Generator class
class Generator(nn.Module):
    def __init__(self, kernels, strides, input_dim=96, image_channels=4, hidden_dim=48):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.generator = nn.Sequential(

            nn.ConvTranspose2d(input_dim * 1, hidden_dim * 8, kernels[0], strides[0]),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernels[1], strides[1]),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernels[2], strides[2]),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 1, kernels[3], strides[3]),
            nn.BatchNorm2d(hidden_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(hidden_dim * 1, image_channels, kernels[4], strides[4]),
            nn.Tanh(),
        )
    
    # Function for completing a forward pass of the generator #
    def forward(self, noise):
        return self.generator(noise.view(len(noise), self.input_dim, 1, 1))
    

# Returns the random noise vector from normal distribution given the dimensions #
def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, device=device)


# Critic class
class Critic(nn.Module):
    def __init__(self, kernels, strides, image_channels=4, hidden_dim=96):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(

            nn.Conv2d(image_channels, hidden_dim * 1, kernels[0], strides[0]),
            nn.BatchNorm2d(hidden_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 1, hidden_dim * 2, kernels[1], strides[1]),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernels[2], strides[2]),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, 1, kernels[3], strides[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    # Function for completing a forward pass of the critic #
    def forward(self, image):
        critic_pred = self.critic(image)
        critic_pred = critic_pred.to('cuda')

        return critic_pred.view(len(critic_pred), -1)
    

# Setup and create paths and directories #
def setup_paths(learning_set_name, dt_string):
    paths = {}
    os.chdir("C:\\Users\\wokol\\Desktop\\KKolis\\Coding BeAll EndAll\\PokemonGAN\\PokemonGAN")
    paths["path_main"] = os.getcwd()
    paths["path_images"] = os.path.join(paths["path_main"], "Images")
    paths["path_original_images"] = os.path.join(paths["path_images"], "OriginalImages")
    paths["path_resized_original_images"] = os.path.join(paths["path_images"], "OriginalResizedImages")
    paths["path_augmented_images"] = os.path.join(paths["path_images"], "PreparedImages")
    paths["path_results"] = os.path.join(paths["path_main"], "Results")
    paths["path_training"] = os.path.join(paths["path_results"], dt_string + "_training_" + learning_set_name)
    paths["path_training_images"] = os.path.join(paths["path_training"], "GeneratedImages")
    paths["path_training_images_grids"] = os.path.join(paths["path_training"], "GeneratedImagesGrids")
    paths["path_training_gifs"] = os.path.join(paths["path_training"], "TrainingGifs")
    paths["path_training_network"] = os.path.join(paths["path_training"], "NeuralNetwork")
    paths["path_training_network_critics"] = os.path.join(paths["path_training_network"], "Critics")
    paths["path_training_network_generators"] = os.path.join(paths["path_training_network"], "Generators")
    paths["path_training_info"] = os.path.join(paths["path_training"], "Info")
    paths["path_csv_database"] = os.path.join(paths["path_resized_original_images"], "PokemonCsv.csv")

    if os.path.isdir(paths["path_resized_original_images"]) is False:
        os.mkdir(paths["path_resized_original_images"])

    if os.path.isdir(paths["path_augmented_images"]) is False:
        os.mkdir(paths["path_augmented_images"])

    if os.path.isdir(paths["path_results"]) is False:
        os.mkdir(paths["path_results"])

    if os.path.isdir(paths["path_training"]) is False:
        os.mkdir(paths["path_training"])

    if os.path.isdir(paths["path_training_images"]) is False:
        os.mkdir(paths["path_training_images"])

    if os.path.isdir(paths["path_training_images_grids"]) is False:
        os.mkdir(paths["path_training_images_grids"])

    if os.path.isdir(paths["path_training_gifs"]) is False:
        os.mkdir(paths["path_training_gifs"])

    if os.path.isdir(paths["path_training_network"]) is False:
        os.mkdir(paths["path_training_network"])

    if os.path.isdir(paths["path_training_network_critics"]) is False:
        os.mkdir(paths["path_training_network_critics"])

    if os.path.isdir(paths["path_training_network_generators"]) is False:
        os.mkdir(paths["path_training_network_generators"])

    if os.path.isdir(paths["path_training_info"]) is False:
        os.mkdir(paths["path_training_info"])

    return paths


# Setup parameters #
def setup_parameters():
    parameters = {}

    # GAN Parameters #
    parameters["n_epochs"] = 500
    parameters["z_dim_generator"] = 128
    parameters["batch_size"] = 4
    parameters["lr"] = 0.0002
    parameters["beta_1"] = 0.9
    parameters["beta_2"] = 0.999
    parameters["c_lambda"] = 10
    parameters["critic_repeats"] = 5

    # Optimization method, 'Adam' or 'RMSprop' #
    parameters["optim"] = 'Adam'

    # GAN Architecture #
    parameters["generator_kernels"] = [4, 4, 4, 4, 6] #[2, 4, 4, 4, 6]
    parameters["generator_strides"] = [2, 2, 2, 2, 2] #[1, 2, 2, 2, 2]
    parameters["critic_kernels"] = [4, 4, 6, 6] #[4, 4, 4, 6]
    parameters["critic_strides"] = [2, 2, 2, 4] #[2, 2, 2, 2]

    # Generated image characteristics #
    parameters["image_dimensions"] = 96
    parameters["image_channels"] = 4

    return parameters


# Setup misc variables #
def setup_misc():
    misc = {}

    now = datetime.now()
    misc["dt_string"] = now.strftime("%Y_%m_%d_%H_%M_%S")
    misc["save_step"] = 1
    misc["generation_nr"] = 32
    misc["learning_set_name"] = "Pokemon"

    return misc


# This is where the magic happens #
def main():
    # Set device and print versions of cuda and cudnn #
    device = soft_control()

    # Setup variables, paths, directories, parameters #
    misc = setup_misc()
    paths = setup_paths(misc["learning_set_name"], misc["dt_string"])
    parameters = setup_parameters()

    # Prepare database, load images, resize, augment etc #
    prepare_database(paths["path_original_images"], paths["path_resized_original_images"], paths["path_augmented_images"], parameters["image_dimensions"])

    # Create main database file by scanning all ball folders #
    create_database_csv(paths["path_resized_original_images"], paths["path_resized_original_images"], "PokemonCsv.csv")

    # Proper training function #
    training(paths, parameters, misc, device)


if __name__ == "__main__":
    main()

