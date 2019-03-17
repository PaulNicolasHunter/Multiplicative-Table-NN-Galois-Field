#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define p printf    // i'm lazy
#define epochs 5    // epochs
#define lrate 0.025 // learning rate

float generate_weight()
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return -0.99 + scale * (1 + 0.99);      /* [min, max] */
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

float act_bro(float y, int der) // activation function Relu
{
    if (y <= 0.0)
    {
        return 0.0;
    }
    else if (der == 0)
    {
        return y;
    }
    else
    {
        return 1.0;
    }
    // if (der == 0)
    // {
    //     return 1 / (1 - exp(-y));
    // }
    // else
    // {
    //     return (1 / (1 - exp(-y))) * (1 - (1 / (1 - exp(-y))));
    // }
    
}
void view_hype(struct neuron *layer)
{
    int i, j;
    for (i = 0; i < 50; i++) // first layer will have 6 inputs on each neuron
    {
        for (j = 0; j < 6; j++)
        {
            p("%f ", layer[i].error); // i = each neuron, j = each incoming input
        }
        p("\n");
        // layer1[i].bias = generate_weight(); // bias
    }
}
void init_network(struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j;

    for (i = 0; i < 50; i++) // first layer will have 6 inputs on each neuron
    {
        for (j = 0; j < 6; j++)
        {
            layer1[i].weight[j] = generate_weight(); // i = each neuron, j = each incoming input
        }
        layer1[i].bias = generate_weight(); // bias
    }

    for (i = 0; i < 70; i++)
    {
        for (j = 0; j < 50; j++)
        {
            layer2[i].weight[j] = generate_weight();
        }
        layer2[i].bias = generate_weight();
    }

    for (i = 0; i < 50; i++)
    {
        for (j = 0; j < 70; j++)
        {
            layer3[i].weight[j] = generate_weight();
        }
        layer3[i].bias = generate_weight();
    }

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 50; j++)
        {
            output_layer[i].weight[j] = generate_weight();
        }
        output_layer[i].bias = generate_weight();
    }
}
void back_prop(struct neuron *layer, int neurons, int weights_num) // operates in that pericular layer
{
    int i, j;

    for (i = 0; i < neurons; i++)
    {
        for (j = 0; j < weights_num; j++)
        {
            layer[i].weight[j] = layer[i].weight[j] + (layer[i].error * lrate * layer[i].value);
            layer[i].bias = layer[i].bias + (layer[i].error * lrate);
        }
    }
    view_hype(layer);
}

void hidden_prop(struct neuron *lay_prev, struct neuron *lay_next, int neu_prev, int neu_next)
{
    int i, j;
    float delta, deltaj;
    for (i = 0; i < neu_prev; i++)
    {
        delta = 0;
        for (j = 0; j < neu_next; j++)
        {
            delta += lay_next[j].error * lay_next[j].weight[i];
        }
        deltaj = delta * act_bro(lay_prev[i].value, 1);
        lay_prev[i].error = deltaj;
        back_prop(lay_prev, 50, 70);
    }
}

void erro_calc(int labels[64][3], int batch_num, struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k;
    float delta, deltaj;

    // for (i = 0; i < 50; i++)
    // {
    //     for (j = 0; j < 6; j++)
    //     {
    //         p("%f ", layer1[i].weight[j]);
    //     }
    //     puts("");
    // }
    for (i = 0; i < 3; i++) // for output layer we directly have the error
    {
        deltaj = (labels[batch_num][i] - output_layer[i].value) * output_layer[i].der; // delta
        output_layer[i].error = deltaj;                                                // error at each output node
    }
    back_prop(output_layer, 3, 50); // updating the weights
    hidden_prop(layer3, output_layer, 50, 3);
    hidden_prop(layer2, layer3, 70, 50);
    hidden_prop(layer1, layer2, 50, 70);
}

void train_network(int logits[64][6], int labels[64][3], struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k, epoch = 0;
    float wsum;

    while (epoch < epochs) // we run this 4 times
    {
        for (i = 0; i < 3; i++) // we have 64 imputs
        {
            for (j = 0; j < 50; j++) // first layer
            {
                wsum = layer1[j].bias;
                for (k = 0; k < 6; k++)
                {
                    wsum = wsum + (layer1[j].weight[k] * (float)logits[i][k]);
                }
                layer1[j].value = act_bro(wsum, 0); // Y-in
                layer1[j].der = act_bro(wsum, 1);   // derivative (Y) = f'(Y-in)
            }

            for (j = 0; j < 70; j++)
            {
                wsum = layer2[j].bias;
                for (k = 0; k < 50; k++)
                {
                    wsum += layer2[j].weight[k] * layer1[k].value;
                }
                layer2[j].value = act_bro(wsum, 0);
                layer2[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < 50; j++)
            {
                wsum = layer3[j].bias;
                for (k = 0; k < 70; k++)
                {
                    wsum += layer3[j].weight[k] * layer2[k].value;
                }
                layer3[j].value = act_bro(wsum, 0);
                layer3[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < 3; j++)
            {
                wsum = output_layer[j].bias;
                for (k = 0; k < 50; k++)
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
        for (j = 0; j < 50; j++)
        {
            wsum = layer1[j].bias;
            for (k = 0; k < 6; k++)
            {
                wsum = wsum + (layer1[j].weight[k] * (float)t[i][k]);
            }
            layer1[j].value = act_bro(wsum, 0); // Y-in
        }

        for (j = 0; j < 70; j++)
        {
            wsum = layer2[j].bias;
            for (k = 0; k < 50; k++)
            {
                wsum += layer2[j].weight[k] * layer1[k].value;
            }
            layer2[j].value = act_bro(wsum, 0);
        }

        for (j = 0; j < 50; j++)
        {
            wsum = layer3[j].bias;
            for (k = 0; k < 70; k++)
            {
                wsum += layer3[j].weight[k] * layer2[k].value;
            }
            layer3[j].value = act_bro(wsum, 0);
        }

        for (j = 0; j < 3; j++)
        {
            wsum = output_layer[j].bias;
            for (k = 0; k < 50; k++)
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
    int logits[64][6] = {
        {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 1}, {0, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1}, {0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 1, 0, 0, 1, 0}, {0, 1, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 1}, {0, 1, 0, 1, 1, 0}, {0, 1, 0, 1, 1, 1}, {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 1, 0}, {0, 1, 1, 0, 1, 1}, {0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 0}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0}, {1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0}, {1, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1}};
    // 64 * 3
    int labels[64][3] = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}};

    layer1 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));      // 50 neuron at first layer
    layer2 = (struct neuron *)(malloc(sizeof(struct neuron) * 70));      // 70
    layer3 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));      // 50
    output_layer = (struct neuron *)(malloc(sizeof(struct neuron) * 3)); // 3 output

    init_network(layer1, layer2, layer3, output_layer);
    train_network(logits, labels, layer1, layer2, layer3, output_layer);
    // test(logits, layer1, layer2, layer3, output_layer);
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