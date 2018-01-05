from data_reader import TutorialDataReader, SequenceDataReader

# Check how many samples are received for each example


def test_receive_all_data():
    def state_initializer():
        return 0

    sequence_size = 50
    batch_size = 32
    dr = TutorialDataReader(readers_count=5, batch_size=batch_size, state_initializer=state_initializer,
                            allow_smaller_batch=True)
    dr.initialize_epoch(randomize=False, sequence_size=sequence_size)
    dr.start_readers()

    for i in range(2):
        examples_time = {}
        examples_data = {}
        try:
            while True:
                batch, time, labels, ids = dr.get_batch()
                for i, (id, b, t) in enumerate(zip(ids, batch, time)):
                    if id not in examples_time.keys():
                        examples_time[id] = []
                        examples_data[id] = []
                    examples_time[id].extend(t.flatten().tolist())
                    examples_data[id].extend(b.flatten().tolist())

                hidden = [0 for _ in range(batch_size)]

                dr.set_states(ids, hidden)

        except IndexError:
            print(examples)
            print('Epoch finished')
            dr.initialize_epoch(randomize=False, sequence_size=sequence_size)

if __name__ == '__main__':
    test_receive_all_data()

