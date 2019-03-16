#include <stdio.h>
#include <stdlib.h>
#define p printf

float generate_weight()
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return -0.56 + scale * (1 + 0.56);      /* [min, max] */
}

struct neuron
{
    long double weight[70];
    long double bias;
    long double value;
};

int activation_(float y)
{
    if (y < 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

void init_network(struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j;
    float bias1 = generate_weight();
    float bias2 = generate_weight();
    float bias3 = generate_weight();
    float bias4 = generate_weight();

    for (i = 0; i < 50; i++)
    {
        for (j = 0; j < 6; j++)
        {
            layer1[i].weight[j] = generate_weight();
        }
        layer1[i].bias = bias1;
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

void train_network(int *logits, int *labels, struct neuron *layer1, struct neuron *layer2, struct neuron *layer3, struct neuron *output_layer)
{
    int i, j, k, l, m, wum;

    for (i = 0; i < 64; i++)
    {
        for (j = 0; j < 50; j++)
        {
            wsum = 0;
            for (k = 0; k < 6; k++)
            {
                wsum += layer1[j].weight[k] * logits[i][k] + layer1[j].bias;
            }
            layer1[j].value = wsum;
        }

        for (j = 0; j < 70; j++)
        {
            wsum = 0;            
            for (k = 0; k < 50; k++)
            {
                wsum += layer2[j].weight[k] * layer1[k].value + layer2[j].bias;
            }
            layer2[j].value = wsum;
        }
    }
}

int main()
{
    struct neuron *layer1, *layer2, *layer3, *output_layer;

    int logits[64][3] = {
        {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 1, 0, 1}, {0, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 1, 0, 1, 0}, {0, 0, 1, 0, 1, 1}, {0, 0, 1, 1, 0, 0}, {0, 0, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 1}, {0, 1, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 1}, {0, 1, 0, 0, 1, 0}, {0, 1, 0, 0, 1, 1}, {0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 1}, {0, 1, 0, 1, 1, 0}, {0, 1, 0, 1, 1, 1}, {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 1}, {0, 1, 1, 0, 1, 0}, {0, 1, 1, 0, 1, 1}, {0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1}, {0, 1, 1, 1, 1, 0}, {0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0}, {1, 0, 0, 1, 0, 1}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0}, {1, 0, 1, 0, 0, 1}, {1, 0, 1, 0, 1, 0}, {1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 0}, {1, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 1, 0}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1}, {1, 1, 0, 0, 1, 0}, {1, 1, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 0, 1}, {1, 1, 0, 1, 1, 0}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 1, 0}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1}};

    int labels[64][3] = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}};

    layer1 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));
    layer2 = (struct neuron *)(malloc(sizeof(struct neuron) * 70));
    layer3 = (struct neuron *)(malloc(sizeof(struct neuron) * 50));
    output_layer = (struct neuron *)(malloc(sizeof(struct neuron) * 3));

    init_network(layer1, layer2, layer3, output_layer);
    process_inputs(logits);
    train_network(logits, labels, layer1, layer2, layer3, output_layer);

    return 0;
}
