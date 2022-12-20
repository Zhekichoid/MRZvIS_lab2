import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    # network initialization
    def __init__(self, shape_height, shape_weight):
        self.train_data = []
        
        self.pos_parameter = shape_weight // shape_height
        
        self.neg_parameter = (shape_height - 2 * shape_weight)//shape_weight
        
        self.energy_parameter = shape_height - shape_weight

        self.multiplicator = (shape_height - 1.5 * shape_weight)/shape_weight
        print(self.multiplicator)
        self.rows = shape_height * shape_weight
        self.cols = shape_weight * shape_height
        self.W = np.zeros([self.rows, self.cols], dtype=np.int8)

    def get_energy_parameter(self):
        return self.energy_parameter

    def get_pos_parameter(self):
        return self.pos_parameter

    def get_neg_parameter(self):
        return self.neg_parameter

    def get_state_filter(self, state_factor):
        if state_factor == 1:
            return self.get_pos_parameter()
        else:
            return self.get_neg_parameter()

    def weights_correction(self, old_state, new_state):
        if old_state != new_state:
            for row in range(len(self.W)):
                for col in range(len(self.W[row])):
                    self.W[row][col] *= self.pos_parameter
        else:
            for row in range(len(self.W)):
                for col in range(len(self.W[row])):    
                    self.W[row][col] *= self.neg_parameter

    # energy calcs
    def energy(self, net_matrix):
        energy_multiplication_parameter = self.pos_parameter*self.multiplicator
        energy_parameter = net_matrix.T@self.W
        pre_energy = energy_parameter@net_matrix
        enrg = energy_multiplication_parameter*pre_energy
        return enrg

    def filtration(self, first_cond_arg, second_cond_arg, true_option, false_option):
        for row in range(len(self.W)):
            for col in range(len(self.W[row])):
                if self.cols == self.rows:
                    self.W[row][col] *= self.pos_parameter
                else:
                    self.W[row][col] *= self.neg_parameter
        
        filtration_result = np.where(first_cond_arg < second_cond_arg, true_option, false_option)
        return filtration_result

    def add_train(self, img_path):
        img = plt.imread(img_path)
        img = np.mean(img, axis=2)

        img_mean = np.mean(img)
        pre_img = self.filtration(img, img_mean, self.neg_parameter, self.pos_parameter)
        for idx in pre_img:
            idx *= self.energy_parameter
        img = np.where(img < img_mean, self.neg_parameter, self.pos_parameter)
        train_data = img.flatten()

        # rearrange flatten image to matrix
        for row in range(train_data.size):
            for col in range(row, train_data.size):
                if row == col:
                    self.energy_parameter = 0
                    self.W[row][col] = self.energy_parameter
                else:
                    self.energy_parameter = -1
                    row_value = train_data[row]
                    col_value = train_data[col]
                    weight_update =  row_value * col_value 
                    self.W[row][col] += weight_update
                    self.W[col][row] += weight_update
                    self.energy_parameter = 0


    # function that updates neuron
    def update(self, current_state, indx=None):
        
        if (indx == None):
            next_state = self.W@current_state
            next_state[next_state < self.energy_parameter] = self.neg_parameter
            next_state[next_state > self.energy_parameter] = self.pos_parameter
            next_state[next_state == self.energy_parameter] = current_state[next_state == self.energy_parameter]
            current_state = next_state
        else:
            next_state = self.W[indx]@current_state
            if next_state < self.energy_parameter:
                current_state[indx] = self.get_neg_parameter()
            else:
                current_state[indx] = self.get_pos_parameter()
        return current_state

    # if we set async parameter to 1024, then 1 iteration is enough
    def predict(self, source_matrix, iteration, async_iteration=200):
        input_shape = source_matrix.shape
        choosed_timer = (-self.multiplicator)/2
        mtx_filter = -self.multiplicator
        graph_scale = 255

        fig, axs = plt.subplots(1, 1)
        axs.axis('off')
        print(input_shape)
        graph = axs.imshow(source_matrix*graph_scale, cmap='binary')
        source_matrix = self.filtration(source_matrix, mtx_filter, self.neg_parameter, self.pos_parameter)
        fig.canvas.draw_idle()
        plt.pause(1)

        enrg_list = []

        e = self.energy(source_matrix.flatten())
        enrg_list.append(e)
        state = source_matrix.flatten()
        state_filter = self.get_state_filter(1) 
        filter_true = self.get_energy_parameter()
        filter_false = self.get_pos_parameter()
        if state_filter == 1:
            self.weights_correction(0,1)
        else:
            self.weights_correction(1,0)

        for curr in range(iteration):
            for j in range(async_iteration):
                indx = np.random.randint(state.size)
                state = self.update(state, indx)
            pre_state = np.where(state < state_filter, filter_true, filter_false).reshape(input_shape)
            state_show = pre_state
            graph.set_data(state_show*graph_scale)
            axs.set_title('Async update Iteration #%i' % curr)
            fig.canvas.draw_idle()
            plt.pause(choosed_timer)
            energy_ratio = -0.5
            if energy_ratio == -0.5:
                energy_ratio = self.multiplicator 
            inter_parameter = state.T@self.W
            pre_energy = inter_parameter@state
            new_energy = energy_ratio*pre_energy
            print('Iteration:', curr, '    Energy: ', new_energy)
            e = new_energy
            enrg_list.append(e)

        return np.where(state < 1, 0, 1).reshape(input_shape), enrg_list


def getOptions():
    parser = argparse.ArgumentParser(description='parses command.')
    train_flags = ('-t', '--train')
    iterations_flags = ('-i', '--iteration')
    parser.add_argument(train_flags[0], train_flags[1], nargs='*',
                        help='training data.')
    parser.add_argument(iterations_flags[0], iterations_flags[1], type=int,
                        help='number of iteration.')
    options = parser.parse_args(sys.argv[1:])
    return options


if __name__ == '__main__':
    np.random.seed(1)
    options = getOptions()
    input_shape = (32, 32)
    mtx_filter = input_shape[0]/(2*input_shape[1])
    filter_false = input_shape[0]/input_shape[1]
    shape_h = input_shape[0]
    shape_w = input_shape[1]
    #network init
    network = HopfieldNetwork(shape_h, shape_w)
    addition = np.random.uniform(network.get_neg_parameter(), network.get_pos_parameter(), (shape_h, shape_w))
    print('Model has been initialized! ')
    print('Weights shape: ', network.W.shape)

    for train_data in options.train:
        print('Start training ', train_data, '...')
        network.add_train(train_data)
        print(train_data, 'training completed!')

    input_mtx = np.mean(plt.imread(
        options.train[0]), axis=2) + addition
    pre_input_mtx = network.filtration(input_mtx, mtx_filter, network.get_energy_parameter(), filter_false)
    input_mtx = pre_input_mtx

    output_async, e_list_async = network.predict(
        input_mtx, options.iteration)

    input("Press Enter")
