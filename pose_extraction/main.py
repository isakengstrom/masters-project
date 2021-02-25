
from FOI_extraction import extract_from_foi_dataset
from extraction_config import DATASET_PATH

if __name__ == "__main__":
    dataset_poses = extract_from_foi_dataset(root_dir=DATASET_PATH)


    print("Number of subjects: {}".format(len(dataset_poses)))
    for subject_poses in dataset_poses:
        print("\tNumber of sequences for : {}".format(len(subject_poses)))
        for sequence_poses in subject_poses:
            print("\t\t Number of angles {}".format(len(sequence_poses)))