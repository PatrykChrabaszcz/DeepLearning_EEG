import click
import re


class NerChallengeDataGenerator:
    class LabelReader:
        def __init__(self, label_file):
            # Index using (subject_id, session_id)
            self.labels = {}

            with open(label_file, 'r') as f:
                # Discard the first header line
                f.readline()
                for line in f.readlines():
                    # S02_Sess01_FB002,1
                    name, label = line.split(',')
                    key = re.findall('S(\d+)_Sess(\d+)_FB.*')[0]
                    label = int(label)
                    if key not in self.labels:
                        self.labels[key] = []
                    self.labels[key].append(label)

        def get_labels(self, subject_id, session_id):
            return self.labels[(subject_id, session_id)]

    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path

    def prepare(self):




        # Train and Validation are located inside the 'train' folder
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            folder_name = 'train'
        elif self.data_type == self.Test_Data:
            folder_name = 'test'
        else:
            raise NotImplementedError('data_type is not from the set {train, validation, test}')

        files = sorted([f.path for f in os.scandir(os.path.join(self.data_path, folder_name)) if f.is_file()])

        # Our test set consists of new subjects, hence we need to split the CV data according to the subjects
        subjects = {}
        subject_indices = []
        for file in files:
            subject_id, session_id = re.findall("Data_S(\d+)_Sess(\d+).csv", file)[0]
            if subject_id not in subjects.keys():
                subjects[subject_id] = []
                subject_indices.append(subject_id)
            subjects[subject_id].append((subject_id, session_id, file))

        # Filter out proper cv fold when validation or train data is used
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            if self.train_on_full:
                assert self.data_type != self.Validation_Data, 'There is no validation data if train_on_full is set ' \
                                                               'to 1'
                validation_subject_indices = None
                train_subject_indices = subject_indices
            else:
                start = int(self.cv_k / self.cv_n * len(subject_indices))
                end = int((self.cv_k + 1) / self.cv_n * len(subject_indices))
                validation_subject_indices = subject_indices[start:end]
                train_subject_indices = subject_indices[:start] + subject_indices[end:]

            if self.data_type == self.Train_Data:
                subject_indices = train_subject_indices
            else:
                subject_indices = validation_subject_indices

        # Read labels
        labels_file = os.path.join(self.data_path, "TrainLabels.csv")
        label_reader = self.LabelReader(labels_file)

        examples = []
        for subject_id in subject_indices:
            subject_id, session_id, file = subjects[subject_id]
            labels = label_reader.get_labels(subject_id, session_id)
            example = self.ExampleInfo(example_id=(subject_id, session_id), data_file=file, labels=labels)
            examples.append(example)

            if self.train_on_full:
                assert self.data_type != self.Validation_Data, 'There is no validation data if train_on_full is set ' \
                                                               'to 1'
                validation_info_dicts = None
                train_info_dicts = info_dicts + info_dicts
            else:
                # Split out the data according to the CV fold
                start = int(self.cv_k / self.cv_n * len(info_dicts))
                end = int((self.cv_k + 1) / self.cv_n * len(info_dicts))
                logger.debug(
                    "Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

                validation_info_dicts = info_dicts[start:end]
                train_info_dicts = info_dicts[:start] + info_dicts[end:]

            if self.data_type == self.Train_Data:
                info_dicts = train_info_dicts
            else:
                info_dicts = validation_info_dicts

        if self.normalization_type == self.normalization_none:
            logger.debug('Will not normalize the data.')
            for info_dict in info_dicts:
                info_dict['mean'] = 0.0
                info_dict['std'] = 1.0
        elif self.normalization_type == self.normalization_separate:
            logger.debug('Will normalize each recording separately.')
        else:
            raise NotImplementedError('This normalization (%s) is not implemented' % self.normalization_type)

        # Create examples
        for i, label in enumerate(labels):
            label_info_dicts = [info_dict for info_dict in info_dicts if info_dict[self.label_type] == label]
            label_info_dicts = label_info_dicts[:self.limit_examples]

            self.examples.append([AnomalyDataReader.ExampleInfo(label_info_dict, self.label_type,
                                                                self.offset_size, self.random_mode,
                                                                self.limit_duration, self.use_augmentation)
                                  for (j, label_info_dict) in enumerate(label_info_dicts)])

            logger.debug('Label %s: Number of recordings %d, Cumulative Length %d' %
                         (label, len(self.examples[i]), sum([e.get_length() for e in self.examples[i]])))

        logger.debug('Number of sequences in the dataset %d' % nested_list_len(self.examples))


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--cache_path', type=click.Path(), required=True)
def main(data_path, cache_path):
    data_generator = NerChallengeDataGenerator(data_path, cache_path)
    data_generator.prepare()

if __name__ == '__main__':
    main()
