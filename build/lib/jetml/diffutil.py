class ArrayDiff:
    def __init__(self, array1, array2, get_compare_value=(lambda i : i), save_both_on_repeat=False):
        self.array1 = array1
        self.array2 = array2
        self.get_compare_value = get_compare_value
        self.save_both_on_repeat = save_both_on_repeat
    
    def diff(self):
        unique_in_1 = []
        unique_in_2 = self.array2[:]
        unique_in_2_compare = [self.get_compare_value(i) for i in unique_in_2]
        repeats = []
        for i in range(len(self.array1)):
            r = self.array1[i]
            r_compare = self.get_compare_value(r)
            if r_compare in unique_in_2_compare:
                remove_index = unique_in_2_compare.index(r_compare)
                if self.save_both_on_repeat:
                    repeats.append((r, unique_in_2[remove_index]))
                else:
                    repeats.append(r)
                del unique_in_2[remove_index]
                del unique_in_2_compare[remove_index]
            else:
                unique_in_1.append(r)
        return Diffs(repeats, unique_in_1, unique_in_2)

class Diffs:
    __slots__ = ("repeats", "unique_in_1", "unique_in_2")
    def __init__(self, repeats, unique_in_1, unique_in_2):
        self.repeats = repeats
        self.unique_in_1 = unique_in_1
        self.unique_in_2 = unique_in_2

