#include <stdio.h>
#include <stdlib.h>
#define p printf    // lazy
#define epochs 5    // epochs
#define lrate 0.025 // learning rate

float generate_weight()
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return -0.56 + scale * (1 + 0.56);      /* [min, max] */
}

struct neuron // structure of each neuron in a layer
{
    float weight[70]; // incoming weights of each neuron
    int weights_num;  // how many inputs each neuron is having: 6 in-case of first layer, 50 incase of second layer
    float bias;       // bias on each node
    float value;      // w*x + v , value of each node
    float der;        // derivative of "value"
    float error;      // error on each node
};

int act_bro(float y, int der) // activation function Relu
{

    if (y < 0)
    {
        return 0;
    }
    else if (!der)
    {
        return y;
    }
    else
    {
        return 1;
    }
}

void init_network(struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j;
    float bias1 = generate_weight(); // each layer will have only one bias
    float bias2 = generate_weight();
    float bias3 = generate_weight();
    float bias4 = generate_weight();

    for (i = 0; i < 50; i++) // first layer will have 6 inputs on each neuron
    {
        for (j = 0; j < 6; j++)
        {
            layer1[i].weight[j] = generate_weight(); // i = each neuron, j = each incoming input
        }
        layer1[i].bias = bias1; // bias
    }

    for (i = 0; i < 70; i++)
    {
        for (j = 0; j < 50; j++)
        {
            layer2[i].weight[j] = generate_weight();
        }
        layer2[i].bias = bias2;
    }

    for (i = 0; i < 50; i++)
    {
        for (j = 0; j < 70; j++)
        {
            layer3[i].weight[j] = generate_weight();
        }
        layer3[i].bias = bias3;
    }

    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 50; j++)
        {
            output_layer[i].weight[j] = generate_weight();
        }
        output_layer[i].bias = bias4;
    }
}

void train_network(int logits[64][6], int labels[64][3], struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k, l, m, wsum;

    while (epochs < 4) // we run this 4 times
    {
        for (i = 0; i < 64; i++) // we have 64 imputs
        {
            for (j = 0; j < 50; j++) // first layer
            {
                wsum = 0;
                for (k = 0; k < 6; k++)
                {
                    wsum = wsum + (layer1[j].weight[k] * (float)logits[i][k] + (float)layer1[j].bias);
                }
                layer1[j].value = act_bro(wsum, 0); // Y-in
                layer1[j].der = act_bro(wsum, 1); // derivative (Y) = f'(Y-in) 
            }

            for (j = 0; j < 70; j++)
            {
                wsum = 0;
                for (k = 0; k < 50; k++)
                {
                    wsum += layer2[j].weight[k] * (float)layer1[k].value + (float)layer2[j].bias;
                }
                layer2[j].value = act_bro(wsum, 0);
                layer2[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < 50; j++)
            {
                wsum = 0;
                for (k = 0; k < 70; k++)
                {
                    wsum += layer3[j].weight[k] * (float)layer2[k].value + (float)layer3[j].bias;
                }
                layer3[j].value = act_bro(wsum, 0);
                layer3[j].der = act_bro(wsum, 1);
            }

            for (j = 0; j < 3; j++)
            {
                wsum = 0;
                for (k = 0; k < 50; k++)
                {
                    wsum += output_layer[j].weight[k] * (float)layer3[k].value + (float)output_layer[j].bias;
                }
            }
            output_layer[j].value = act_bro(wsum, 0);
            output_layer[j].der = act_bro(wsum, 1);

            erro_calc(labels, i, layer1, layer2, layer3, output_layer);
        }
    }
}

void erro_calc(int labels[64][3], int batch_num, struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k;
    float delta, deltaj;

    for (i = 0; i < 3; i++)  // for output layer we directly have the error
    {
        delta = (labels[batch_num][i] - output_layer[i].value) * output_layer[i].der; // delta
        output_layer[i].error = delta; // error at each output node
        back_prop(output_layer, 3, 50, delta); // just updating the weights
    }

    for (i = 0; i < 50; i++)
    {
        delta = 0;
        for (j = 0; j < 3; j++)
        {
            delta += output_layer[j].error * layer3[i].weight[j];
        }

        deltaj = delta * act_bro(layer3[i].value, 1);
    }

    // back_prop(layer3, 50, 70, error);
    // back_prop(layer2, 70, 50, error);
    // back_prop(layer1, 50, 6, error);
}

void back_prop(struct neuron *layer, int neurons, int weights_num, float error)
{
    int i, j;

    for (i = 0; i < neurons; i++)
    {
        for (j = 0; j < weights_num; j++)
        {
            layer[i].weight[j] = layer[i].weight[j] * layer[i].value * lrate;
            layer[i].bias = layer[i].value * lrate;
        }
    }
}

int main()
{
    struct neuron *layer1, *layer2, *layer3, *output_layer;
    // 64 * 6
    int logits[64][6] = {
        {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 1}, {0, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1}, {0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 1, 0, 0, 1, 0}, {0, 1, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 1}, {0, 1, 0, 1, 1, 0}, {0, 1, 0, 1, 1, 1}, {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 1, 0}, {0, 1, 1, 0, 1, 1}, {0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 0}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0}, {1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0}, {1, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1}};
    // 64 * 3
    int labels[64][3] = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}};

    layer1 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));
    layer2 = (struct neuron *)(malloc(sizeof(struct neuron) * 70));
    layer3 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));
    output_layer = (struct neuron *)(malloc(sizeof(struct neuron) * 3));

    init_network(layer1, layer2, layer3, output_layer);
    train_network(logits, labels, layer1, layer2, layer3, output_layer);
    // write the weights and test
    return 0;
}
