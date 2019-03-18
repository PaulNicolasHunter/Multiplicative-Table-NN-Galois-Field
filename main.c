#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define p printf     // i'm lazy
#define epochs 15   // epochs
#define lrate 0.025 // learning rate
#define size1 2
#define size2 5
#define size3 4
#define input_size 6
#define output_size 3


float generate_weight()
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return -0.98 + scale * (1.2 + 0.98);    /* [min, max] */
}
// weights_num and bias and make it 2b and ask for neurons
struct neuron // structure of each neuron in a layer
{
    float weight[70]; // incoming weights of each neuron
    // int weights_num;  // how many inputs each neuron is having: 6 in-case of first layer, 50 incase of second layer
    float bias;  // bias on each node
    float value; // w*x + v , value of each node
    float der;   // derivative of "value"
    float error; // error on each node
};

float act_bro(float y, int der) // activation function LeakyRelu
{
    if (y <= 0.0)
    {
        return 0.01 * y;
    }
    else if (der == 0)
    {
        return y;
    }
    else
    {
        return 1.0;
    }
}

void init_network(struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j;

    for (i = 0; i < size1; i++) // first layer will have 6 inputs on each neuron
    {
        for (j = 0; j < input_size; j++)
        {
            layer1[i].weight[j] = generate_weight(); // i = each neuron, j = each incoming input
        }
        layer1[i].bias = generate_weight(); // bias
    }

    for (i = 0; i < size2; i++)
    {
        for (j = 0; j < size1; j++)
        {
            layer2[i].weight[j] = generate_weight();
        }
        layer2[i].bias = generate_weight();
    }

    for (i = 0; i < size3; i++)
    {
        for (j = 0; j < size2; j++)
        {
            layer3[i].weight[j] = generate_weight();
        }
        layer3[i].bias = generate_weight();
    }

    for (i = 0; i < output_size; i++)
    {
        for (j = 0; j < size3; j++)
        {
            output_layer[i].weight[j] = generate_weight();
        }
        output_layer[i].bias = generate_weight();
    }
}
void back_prop(struct neuron *layer, int neurons, int weights_incoming) // operates in that pericular layer
{
    int i, j;

    for (i = 0; i < neurons; i++)
    {
        for (j = 0; j < weights_incoming; j++)
        {
            layer[i].weight[j] = layer[i].weight[j] + (layer[i].error * lrate * layer[i].value);
            layer[i].bias = layer[i].bias + (layer[i].error * lrate);
        }
    }
}

void hidden_prop(struct neuron *current_lay, struct neuron *lay_next, int neu_curr, int neu_next, int weights_incoming)
{
    int i, j;
    float delta, deltaj;
    for (i = 0; i < neu_curr; i++)
    {
        delta = 0;
        for (j = 0; j < neu_next; j++)
        {
            delta += lay_next[j].error * lay_next[j].weight[i];
        }
        deltaj = delta * act_bro(current_lay[i].value, 1);
        current_lay[i].error = deltaj;
    }
    back_prop(current_lay, neu_curr, weights_incoming);
}

void erro_calc(int labels[64][3], int batch_num, struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k;
    float delta, deltaj;

    for (i = 0; i < 3; i++) // for output layer we directly have the error
    {
        deltaj = (labels[batch_num][i] - output_layer[i].value) * output_layer[i].der; // delta
        output_layer[i].error = deltaj;                                                // error at each output node
        // p(" value=%f lab=%d error=%f ", output_layer[i].value, labels[batch_num][i], output_layer[i].error);
        // p("\n");
    }
    back_prop(output_layer, 3, size1); // updating the weights
    hidden_prop(layer3, output_layer, size3, 3, size2);
    hidden_prop(layer2, layer3, size2, size1, size3);
    hidden_prop(layer1, layer2, size1, size2, 6);
}

void train_network(int logits[64][6], int labels[64][3], struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k, epoch = 0;
    float wsum;

    while (epoch < epochs) // we run this 4 times
    {
        for (i = 0; i < 64; i++) // we have 64 imputs
        {
            // p("\n%d\n", i);

            for (j = 0; j < size1; j++) // first layer
            {
                wsum = layer1[j].bias;
                for (k = 0; k < input_size; k++)
                {
                    wsum = wsum + (layer1[j].weight[k] * (float)logits[i][k]);
                }
                layer1[j].value = act_bro(wsum, 0); // Y-in
                layer1[j].der = act_bro(wsum, 1);   // derivative (Y) = f'(Y-in)
            }

            for (j = 0; j < size2; j++)
            {
                wsum = layer2[j].bias;
                for (k = 0; k < size1; k++)
                {
                    wsum += layer2[j].weight[k] * layer1[k].value;
                }
                layer2[j].value = act_bro(wsum, 0);
                layer2[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < size3; j++)
            {
                wsum = layer3[j].bias;
                for (k = 0; k < size2; k++)
                {
                    wsum += layer3[j].weight[k] * layer2[k].value;
                }
                layer3[j].value = act_bro(wsum, 0);
                layer3[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < output_size; j++)
            {
                wsum = output_layer[j].bias;
                for (k = 0; k < size3; k++)
                {
                    wsum += output_layer[j].weight[k] * layer3[k].value;
                }
                output_layer[j].value = act_bro(wsum, 0);
                output_layer[j].der = act_bro(wsum, 1);
            }

            erro_calc(labels, i, layer1, layer2, layer3, output_layer);
        }
        // if (epoch % 10 == 0)
        // {
        p("%d / %d\n", epoch, epochs);
        // }
        epoch += 1;
    }
}

void test(int t[64][6], struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int wsum, i, j, k;
    for (i = 0; i < 64; i++)
    {
        for (j = 0; j < size1; j++)
        {
            wsum = layer1[j].bias;
            for (k = 0; k < 6; k++)
            {
                wsum = wsum + (layer1[j].weight[k] * (float)t[i][k]);
            }
            layer1[j].value = act_bro(wsum, 0); // Y-in
        }

        for (j = 0; j < size2; j++)
        {
            wsum = layer2[j].bias;
            for (k = 0; k < size1; k++)
            {
                wsum += layer2[j].weight[k] * layer1[k].value;
            }
            layer2[j].value = act_bro(wsum, 0);
        }

        for (j = 0; j < size3; j++)
        {
            wsum = layer3[j].bias;
            for (k = 0; k < size2; k++)
            {
                wsum += layer3[j].weight[k] * layer2[k].value;
            }
            layer3[j].value = act_bro(wsum, 0);
        }

        for (j = 0; j < output_size; j++)
        {
            wsum = output_layer[j].bias;
            for (k = 0; k < size3; k++)
            {
                wsum += output_layer[j].weight[k] * layer3[k].value;
            }
            output_layer[j].value = act_bro(wsum, 0);
            p("%f ", output_layer[j].value);
        }
        puts("");
    }
}

int main()
{
    struct neuron *layer1, *layer2, *layer3, *output_layer;
    // 64 * 6
    int i, j;
    int logits[64][input_size] = {
        {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 1}, {0, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1}, {0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 1, 0, 0, 1, 0}, {0, 1, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 1}, {0, 1, 0, 1, 1, 0}, {0, 1, 0, 1, 1, 1}, {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 1, 0}, {0, 1, 1, 0, 1, 1}, {0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 0}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0}, {1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0}, {1, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1}};
    // 64 * 3
    int labels[64][output_size] = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}};

    layer1 = (struct neuron *)(malloc(sizeof(struct neuron) * size1));      // 5 neuron at first layer
    layer2 = (struct neuron *)(malloc(sizeof(struct neuron) * size2));      // 70
    layer3 = (struct neuron *)(malloc(sizeof(struct neuron) * size3));      // 50
    output_layer = (struct neuron *)(malloc(sizeof(struct neuron) * output_size)); // 3 output

    init_network(layer1, layer2, layer3, output_layer);
    train_network(logits, labels, layer1, layer2, layer3, output_layer);
    test(logits, layer1, layer2, layer3, output_layer);
    // write the weights and test
    // puts("*-*-*");
    // for (i = 0; i < 50; i++)
    // {
    //     for (j = 0; j < 6; j++)
    //     {
    //         p("%f ", layer1[i].weight[j]);
    //     }
    //     puts("*-*-");
    // }
    return 0;
}
