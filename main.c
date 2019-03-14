#include <stdio.h>
#include <stdlib.h>
#define p printf

float generate_weight()
{
    float scale = rand() / (float)RAND_MAX; /* [0, 1.0] */
    return 0.02 + scale * (1 - 0.02);        /* [min, max] */
}

struct neuron
{
    float weight[70];
    float bias;
    float value;
} ;

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

void init_network(struct neuron *layer1, struct neuron *layer2, struct neuron *layer3)
{
    int i;
    layer1->bias = generate_weight();
    for (i = 0; i < 50; i++)
    {
        layer1->weight[i] = generate_weight();
    }

    layer2->bias = generate_weight();
    for (i = 0; i < 70; i++)
    {
        layer2->weight[i] = generate_weight();
    }

    layer3->bias = generate_weight();
    for (i = 0; i < 50; i++)
    {
        layer3->weight[i] = generate_weight();
    }
}

void train_network(logits, labels, layer1, layer2, layer3) {

    

}

int main()
{
    // struct neuron neu;
    struct neuron *layer1, *layer2, *layer3;
    // layer1 = neu;
    // layer2 = neu;
    // layer3 = neu;

    float ws[6][50], b[50];
    float ws1[50][70], b1[70];
    float ws2[90][50], b2[50];
    float ws3[50][3], b3[3];
    int logits[3][3], labels[3];

    int logits[8][3] = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

    int labels[64][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}, {0, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 1}, {1, 0, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}};

    layer1 = (struct neuron *)(malloc(sizeof(layer1) * 250));
    layer2 = (struct neuron *)(malloc(sizeof(layer1) * 490));
    layer3 = (struct neuron *)(malloc(sizeof(layer1) * 250));

    init_network(layer1, layer2, layer3);
    train_network(logits, labels, layer1, layer2, layer3);

    return 0;
}
