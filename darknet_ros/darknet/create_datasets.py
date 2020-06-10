import argparse
from pathlib import Path
import sys
import random
import os
import shutil


# Get the indices for the datasets
def get_indices(num_examples, train_pct):

    indices = list(range(0, num_examples))
    random.shuffle(indices)

    num_train = int(num_examples*train_pct/100)
    num_valid = int(num_examples*(100-train_pct)/100/2)

    train_indices = indices[:num_train]
    valid_indices = indices[num_train:num_train+num_valid]
    test_indices = indices[num_train+num_valid:]

    return train_indices, valid_indices, test_indices


# Copy images and label files to destination
def copy_dataset(list_indices, image_files, labels_dir, destination):

    for idx in list_indices:
        shutil.copy(str(image_files[idx]), str(destination.joinpath(image_files[idx].name)))
        label_filename = image_files[idx].stem + ".txt"
        label_filepath = labels_dir.joinpath(label_filename)
        shutil.copy(str(label_filepath), str(destination.joinpath(label_filename)))

    print("Copied {} images and labels to {}".format(len(list_indices), destination))


def main():
    parser = argparse.ArgumentParser(description="Create datasets to train YOLO on Duckietown images")
    parser.add_argument("datadir", help="Directory containing the subfolders frames and labels", type=str)
    parser.add_argument("outputdir", help="Directory that will contain the datasets", type=str)
    parser.add_argument("train_pct", help="Percentage to use to create training dataset", type=int)

    args = parser.parse_args()

    datadir = Path(args.datadir)
    outputdir =Path(args.outputdir)
    train_pct = args.train_pct

    #if not datadir.is_dir():
       # print("{} is not a directory".format(datadir))
       #sys.exit(1)

    #if not outputdir.is_dir():
        #print("{} is not a directory".format(outputdir))
       # sys.exit(1)

    # Make sure there are no directories trainset, validset and testset in output directory
    trainset_dir = outputdir.joinpath("trainset")
    validset_dir = outputdir.joinpath("validset")
    testset_dir = outputdir.joinpath("testset")
    print(trainset_dir)

    if trainset_dir.is_dir() or validset_dir.is_dir() or testset_dir.is_dir():
        print("There already exists at least one of trainset, validset, testset directories in {}".format(outputdir))
        sys.exit(1)

    if train_pct < 80 or train_pct > 90:
        print("The training percentage must be between 80 and 90 but you entered {}".format(train_pct))
        sys.exit(1)

    frames_dir = datadir.joinpath("frames")
    labels_dir = datadir.joinpath("labels")

    if not frames_dir.is_dir() or not labels_dir.is_dir():
        print("We expect two subfolders inside {}: frames and labels, but at least one is missing".format(datadir))
        sys.exit(1)

    image_files = list(frames_dir.glob('*.jpg'))
    labels_files = list(labels_dir.glob('*.txt'))

    num_image_files = len(image_files)
    num_labels_files = len(labels_files)

    print("Found {} jpg files and {} txt files, which we consider labels".format(num_image_files, num_labels_files))

    if num_image_files != num_labels_files:
        print("[ERROR] The number of jpg files and txt files must be equal")
        sys.exit(1)

    train_indices, valid_indices, test_indices = get_indices(num_image_files, train_pct)

    # Make the directories
    os.mkdir(str(trainset_dir))
    os.mkdir(str(validset_dir))
    os.mkdir(str(testset_dir))

    # Do the copying
    copy_dataset(train_indices, image_files, labels_dir, trainset_dir)
    copy_dataset(valid_indices, image_files, labels_dir, validset_dir)
    copy_dataset(test_indices, image_files, labels_dir, testset_dir)


if __name__ == "__main__":
    main()
