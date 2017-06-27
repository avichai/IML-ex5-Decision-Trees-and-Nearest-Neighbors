import numpy as np
from matplotlib import pyplot as plt


def get_majority(labels):
    if labels[labels == 1].size >= labels[labels == 0].size:
        return 1
    return 0


def cost(a):
    return min((a, 1 - a))


def gain(train_set, labels, ind):
    x_i = train_set[:, ind]

    p_y = np.count_nonzero(labels) / labels.size

    total_cost = cost(p_y)

    for i in range(3):
        s_v_dis = x_i[x_i == i].size / x_i.size
        p_s_v = labels[x_i == i].size / labels.size
        cost_s_v = s_v_dis * cost(p_s_v)

        total_cost -= cost_s_v

    return total_cost


def get_ind_max_from_gain(train_set, labels, feature_coord):
    max_gain = 0
    max_ind = 0
    for i in range(feature_coord.size):
        ind = feature_coord[i]
        cur_gain = gain(train_set, labels, ind)
        if cur_gain > max_gain:
            max_gain = cur_gain
            max_ind = ind

    if max_gain > 0:
        return max_ind
    return None


def get_examples(train_set, labels, ind_max):
    return ((train_set[train_set[:, ind_max] == 0],
             labels[train_set[:, ind_max] == 0]),
            (train_set[train_set[:, ind_max] == 1],
             labels[train_set[:, ind_max] == 1]),
            (train_set[train_set[:, ind_max] == 2],
             labels[train_set[:, ind_max] == 2]))


def id3(train_set, labels, feature_coord, available_depth):
    if np.all(labels):
        return Node(1, None, None, None, True, 0)
    if np.count_nonzero(labels) == 0:
        return Node(0, None, None, None, True, 0)

    if feature_coord.size == 0 or available_depth == 0:
        majority_val = get_majority(labels)
        return Node(majority_val, None, None, None, True, 0)
    ind_max = get_ind_max_from_gain(train_set, labels, feature_coord)
    if ind_max is None:
        majority_val = get_majority(labels)
        return Node(majority_val, None, None, None, True, 0)
    new_feat = feature_coord[feature_coord != ind_max]
    split_examples = get_examples(train_set, labels, ind_max)
    node = Node(None,
                id3(split_examples[0][0], split_examples[0][1], new_feat,
                    available_depth-1),
                id3(split_examples[1][0], split_examples[1][1], new_feat,
                    available_depth-1),
                id3(split_examples[2][0], split_examples[2][1], new_feat,
                    available_depth-1),
                False, ind_max)
    return node


class Node:
    value_leaf = 0
    left_child = None
    mid_child = None
    right_child = None
    is_leaf = False
    ind_attrib = 0

    def __init__(self, val, left_child, mid_child, right_child,
                 leaf, ind_attrib):
        self.value_leaf = val
        self.left_child = left_child
        self.right_child = right_child
        self.mid_child = mid_child
        self.is_leaf = leaf
        self.ind_attrib = ind_attrib

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_mid_child(self, mid_child):
        self.mid_child = mid_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def get_left_child(self):
        return self.left_child

    def get_mid_child(self):
        return self.mid_child

    def get_right_child(self):
        return self.right_child

    def get_val(self):
        return self.value_leaf

    def get_is_leaf(self):
        return self.is_leaf

    def get_ind_attrib(self):
        return self.ind_attrib


def test_example(tree, example):
    if tree is None:
        return

    while True:
        # print(tree.get_ind_attrib())
        if tree.is_leaf:
            return tree.get_val()
        if example[tree.get_ind_attrib()] == 0:
            tree = tree.get_left_child()
        elif example[tree.get_ind_attrib()] == 1:
            tree = tree.get_mid_child()
        else:
            tree = tree.get_right_child()


def test_examples(tree, train, labels):
    count = 0
    for i in range(train.shape[0]):
        if test_example(tree, train[i]) != labels[i]:
            count += 1
    return count / labels.size


def disp_errors(errors_test, errors_train):
    print(errors_train)
    print(errors_test)
    plt.figure()
    x = np.array(range(15))
    plt.subplot(121)
    plt.title('error train')
    plt.xlabel('depth of tree')
    plt.ylabel('error on train set')
    plt.plot(x, errors_train)
    plt.subplot(122)
    plt.title('error test')
    plt.xlabel('depth of tree')
    plt.ylabel('error on validation set')
    plt.plot(x, errors_test)

    plt.show()


def main():
    text_file = open("train.txt")

    lines = text_file.read().split('\n')
    lines = [w.split(' ') for w in lines]
    lines = [x for x in lines if x != ['']]
    # print(lines[0])

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            lines[i][j] = lines[i][j].replace('republican.', '0')
            lines[i][j] = lines[i][j].replace('democrat.', '1')
            lines[i][j] = lines[i][j].replace('n', '0')
            lines[i][j] = lines[i][j].replace('y', '1')
            lines[i][j] = lines[i][j].replace('u', '2')

    # print(lines[0])
    a = np.array(lines, dtype='int32')

    train = a[:, :-1]
    labels = a[:, -1:]

    text_val = open("validation.txt")

    lines_test = text_val.read().split('\n')
    lines_test = [w.split(' ') for w in lines_test]
    lines_test = [x for x in lines_test if x != ['']]
    # print(lines[0])

    for i in range(len(lines_test)):
        for j in range(len(lines_test[i])):
            lines_test[i][j] = lines_test[i][j].replace('republican.', '0')
            lines_test[i][j] = lines_test[i][j].replace('democrat.', '1')
            lines_test[i][j] = lines_test[i][j].replace('n', '0')
            lines_test[i][j] = lines_test[i][j].replace('y', '1')
            lines_test[i][j] = lines_test[i][j].replace('u', '2')

    # print(lines[0])
    a = np.array(lines_test, dtype='int32')

    test = a[:, :-1]
    labels_test = a[:, -1:]

    errors_train = []
    errors_test = []

    for i in range(15):
        tree = id3(train, labels, np.array(np.arange(train.shape[1])), i)
        errors_train.append(test_examples(tree, train, labels))
        errors_test.append(test_examples(tree, test, labels_test))

    disp_errors(errors_test, errors_train)



if __name__ == '__main__':
    main()
