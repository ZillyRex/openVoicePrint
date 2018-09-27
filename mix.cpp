#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IN_FILE1 "1.wav"
#define IN_FILE2 "2.wav"
#define OUT_FILE "remix.pcm"

#define SIZE_AUDIO_FRAME (2)

void Mix(char sourseFile[10][SIZE_AUDIO_FRAME], int number, char *objectFile)
{
    //归一化混音
    int const MAX = 32767;
    int const MIN = -32768;

    double f = 1;
    int output;
    int i = 0, j = 0;
    for (i = 0; i < SIZE_AUDIO_FRAME / 2; i++)
    {
        int temp = 0;
        for (j = 0; j < number; j++)
        {
            temp += *(short *)(sourseFile[j] + i * 2);
        }
        output = (int)(temp * f);
        if (output > MAX)
        {
            f = (double)MAX / (double)(output);
            output = MAX;
        }
        if (output < MIN)
        {
            f = (double)MIN / (double)(output);
            output = MIN;
        }
        if (f < 1)
        {
            f += ((double)1 - f) / (double)32;
        }
        *(short *)(objectFile + i * 2) = (short)output;
    }
}

#define ARCH_ADD(p, a) ((p) += (a))
void alsa_mix_16(short data1, short data2, short *date_mix)
{
    int sample, old_sample, sum;
    sample = data1;
    old_sample = data2;
    sum = data2;

    if (*date_mix == 0)
        sample -= old_sample;
    ARCH_ADD(sum, sample);
    do
    {
        old_sample = sum;
        if (old_sample > 0x7fff)
            sample = 0x7fff;
        else if (old_sample < -0x8000)
            sample = -0x8000;
        else
            sample = old_sample;
        *date_mix = sample;
    } while (0);
}

int main()
{
    FILE *fp1, *fp2, *fpm;
    fp1 = fopen(IN_FILE1, "rb");
    fp2 = fopen(IN_FILE2, "rb");
    fpm = fopen(OUT_FILE, "wb");

    short data1, data2, date_mix = 0;
    int ret1, ret2;
    char sourseFile[10][2];

    while (1)
    {
        ret1 = fread(&data1, 2, 1, fp1);
        ret2 = fread(&data2, 2, 1, fp2);
        *(short *)sourseFile[0] = data1;
        *(short *)sourseFile[1] = data2;

        if (ret1 > 0 && ret2 > 0)
        {
            Mix(sourseFile, 2, (char *)&date_mix);
            // alsa_mix_16(data1, data2, &date_mix);

            /* if( data1 < 0 && data2 < 0)
			   date_mix = data1+data2 - (data1 * data2 / -(pow(2,16-1)-1));
			   else
			   date_mix = data1+data2 - (data1 * data2 / (pow(2,16-1)-1));*/

            if (date_mix > pow(2, 16 - 1) || date_mix < -pow(2, 16 - 1))
                printf("mix error\n");
        }
        else if ((ret1 > 0) && (ret2 == 0))
        {
            date_mix = data1;
        }
        else if ((ret2 > 0) && (ret1 == 0))
        {
            date_mix = data2;
        }
        else if ((ret1 == 0) && (ret2 == 0))
        {
            break;
        }
        fwrite(&date_mix, 2, 1, fpm);
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fpm);
    printf("Done!\n");
}