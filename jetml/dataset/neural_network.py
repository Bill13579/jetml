from jetml.utils import array_to_vector

class IODataset:
    def __init__(self, dataset=[], label=None):
        self.dataset = dataset
        self.label = label
    
    def __str__(self):
        string = ""
        label = None
        if self.label is not None:
            label = [[l for block in self.label for l in block]]
            input_labels = self.label[0]
            output_labels = self.label[1]
            string += "Input: " + ", ".join(input_labels) + "\n"
            string += "Output: " + ", ".join(output_labels) + "\n"
        data = [[tmp[0] for c in td for tmp in c.tolist()] for td in self.dataset]
        data = [[str(i) for i in d] for d in data]
        if label is None:
            line_number = 0
            both = data
        else:
            line_number = -1
            both = label + data
        both_unwrapped = [i for d in both for i in d]
        both_unwrapped_len = [len(i) for i in both_unwrapped]
        alignment = max(both_unwrapped_len)+2
        for l in range(len(both)):
            line = both[l]
            row ="{{:{}}}".format(alignment) * len(line[:])
            final_row = row.format(*line[:])
            line_number += 1
            if line_number == 0:
                final_row = "\t" + final_row
            else:
                final_row = str(line_number) + "\t" + final_row
            string += final_row + "\n"
        return string

    def add(self, input_data, expected_output):
        self.dataset.append((input_data, expected_output))
    
    def add_all(self, *io_sets):
        for s in io_sets:
            self.dataset.append(s)
    
    def to_batches(self, batch_size):
        batches = []
        if len(self.dataset) < batch_size:
            raise IODataset.DataDeficientException("Not enough data for batches with size " + str(batch_size))
        for i in range(0, len(self.dataset), batch_size):
            batches.append(IODataset(self.dataset[i:i+batch_size], self.label))
        return batches

    @staticmethod
    def from_array(array):
        dataset = []
        for data in array:
            dataset.append((array_to_vector(data[0]), array_to_vector(data[1])))
        return IODataset(dataset)

    class DataDeficientException(Exception):
        pass

