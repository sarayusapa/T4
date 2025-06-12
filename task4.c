#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_DATA 5762
#define THRESHOLD_LSA 0.6
#define THRESHOLD_IR 70.0


int dataCount = 0;


typedef struct
{
   float lsa[5];
   float ir;
} sensor;


sensor input_raw[MAX_DATA];
sensor input_binary[MAX_DATA];


//text file to readings
#include <ctype.h>

void readRawData(const char *file)
{
    FILE *fl = fopen(file, "r");
    if (!fl) {
        perror("Error opening file");
        exit(1);
    }

    char line[256];  // Allow enough space for a full line
    while (fgets(line, sizeof(line), fl)) {
        int count = 0;
        char *ptr = line;

        while (*ptr && count < 6) {

            while (*ptr && !(isdigit(*ptr) || *ptr == '.')) {
                ptr++;
            }

            float val;
            int len;
            if (sscanf(ptr, "%f%n", &val, &len) == 1) {
                if (count < 5) {
                    input_raw[dataCount].lsa[count] = val;
                } else {
                    input_raw[dataCount].ir = val;
                }
                count++;
                ptr += len;
            } else {
                break; 
            }
        }

        if (count == 6) {
            dataCount++;
            if (dataCount >= MAX_DATA) break;
        } else {
            fprintf(stderr, "Invalid line at %d: \"%s\"\n", dataCount + 1, line);
        }
    }

    fclose(fl);
}




//thresholding lsa and ir readings
void convertToBinary(sensor input_raw[], int count)
{
   for (int i = 0; i < count; i++)
   {
       for (int j = 0; j < 5; j++)
       {
           input_binary[i].lsa[j] = (input_raw[i].lsa[j] >= THRESHOLD_LSA) ? 1.0 : 0;
       }
       input_binary[i].ir = (input_raw[i].ir >= THRESHOLD_IR) ? 1.0 : 0;
   }
}


//defines all kinds of junctions
typedef enum
{
   ONLY_LEFT,
   ONLY_RIGHT,
   STRAIGHT,
   PLUS,
   STRAIGHT_AND_RIGHT,
   STRAIGHT_AND_LEFT,
   DEAD_END
} junction;


//defines all types of directions
typedef enum
{
   NORTH,
   SOUTH,
   EAST,
   WEST
} direction;


//defines coordinate values
typedef struct
{
   int x, y;
   junction junc;
   direction dir;
} coord;
coord coo[MAX_DATA];


//converting lsa values to junction types
junction road(float lsa[5], float ir)
{
   if (ir == 1)
       return DEAD_END;


   if (lsa[0] && lsa[1] && lsa[2] && lsa[3] && lsa[4])
       return PLUS;
   else if (lsa[0] && lsa[1] && lsa[2] && lsa[3])
       return ONLY_LEFT;
   else if (lsa[1] && lsa[2] && lsa[3] && lsa[4])
       return ONLY_RIGHT;
   else 
       return STRAIGHT;
}


// changing direction after encountering junction
direction direc(direction dir, junction j)
{
   if (j == STRAIGHT || j == PLUS)
       return dir;
   if (j == DEAD_END) //u turn
   {
       switch (dir)
       {
       case NORTH:
           return SOUTH;
       case SOUTH:
           return NORTH;
       case EAST:
           return WEST;
       case WEST:
           return EAST;
       }
   }
   if (dir == NORTH) 
   {
       if (j == ONLY_RIGHT || j == STRAIGHT_AND_RIGHT)
           return EAST; //north + right = east
       else if (j == ONLY_LEFT || j == STRAIGHT_AND_LEFT)
           return WEST; //north + left = west
   }
   else if (dir == SOUTH)
   {
       if (j == ONLY_RIGHT || j == STRAIGHT_AND_RIGHT)
           return WEST; //south + right = west
       else if (j == ONLY_LEFT || j == STRAIGHT_AND_LEFT)
           return EAST; //south + left = east
   }
   else if (dir == EAST)
   {
       if (j == ONLY_RIGHT || j == STRAIGHT_AND_RIGHT)
           return SOUTH; //east + right = south
       else if (j == ONLY_LEFT || j == STRAIGHT_AND_LEFT)
           return NORTH; //east + left = north
   }
   else if (dir == WEST)
   {
       if (j == ONLY_RIGHT || j == STRAIGHT_AND_RIGHT)
           return NORTH; //west + right = north
       else if (j == ONLY_LEFT || j == STRAIGHT_AND_LEFT)
           return SOUTH; //west + left = south
   }
}



int convertLSA_xy(sensor input_binary[], coord coo[]) 
{
    coo[0].x = 0;
    coo[0].y = 0;
    coo[0].junc = STRAIGHT;
    coo[0].dir = SOUTH;

    junction j;
    direction d = SOUTH; // since north is reference, initial trajectory = north -> south
    int ctr = 1;

    for (int i = 1; i < dataCount && ctr < MAX_DATA; i++)
    {
        j = road(input_binary[i].lsa, input_binary[i].ir);

        if (j == DEAD_END) {
            coordinate_update(d, j, ctr);
            ctr++;
            i++;

            // Skip until next valid junction
            while (i < dataCount && road(input_binary[i].lsa, input_binary[i].ir)==DEAD_END) {
                i++;
            }
            i=i-1;

            if (i >= dataCount) 
            return ctr; // Bounds check after skip

            d = direc(d,j);
            coordinate_update(d, j, ctr);
            ctr++;
            i++;

            if (i >= dataCount) 
            return ctr; // Bounds check 
        }

        d = direc(d, j);

        if (j != STRAIGHT) 
        {
            int j_first = j;
            int mid = ctr + 2; // we assumed thickness of road is covered in 5 readings, so middle appears after 2 readings

            while (ctr <= mid && ctr < MAX_DATA && i < dataCount) 
            {
                if (ctr == mid) {
                    d = direc(d, j);
                    coo[ctr].junc = j;
                } else {
                    coo[ctr].junc = STRAIGHT;
                }
                coordinate_update(d, j, ctr);
                coo[ctr].dir = d;
                ctr++;
                i++;
            }

            while (i < dataCount && road(input_binary[i].lsa, input_binary[i].ir) == j_first) 
            {
                i++;
            }

            if (i >= dataCount || ctr >= MAX_DATA) 
            return ctr;
        }

        j = road(input_binary[i].lsa, input_binary[i].ir);
        d = direc(d, j);
        if (ctr < MAX_DATA) {
            coo[ctr].junc = j;
            coordinate_update(d, j, ctr);
            coo[ctr].dir = d;
            ctr++;
        }
    }

    return ctr;
}


//updating xy coordinates based on current direction
void coordinate_update(direction d, junction j, int ctr) { 
    coo[ctr].dir = d;
    coo[ctr].junc = j;
    switch (d)
       {
       case NORTH:
           coo[ctr].y = coo[ctr - 1].y + 1;
           coo[ctr].x = coo[ctr - 1].x;
           break;
       case SOUTH:
           coo[ctr].y = coo[ctr - 1].y - 1;
           coo[ctr].x = coo[ctr - 1].x;
           break;
       case EAST:
           coo[ctr].y = coo[ctr - 1].y;
           coo[ctr].x = coo[ctr - 1].x + 1;
           break;
       case WEST:
           coo[ctr].y = coo[ctr - 1].y;
           coo[ctr].x = coo[ctr - 1].x - 1;
           break;
       }
}


// update junctions when an xy coordinate is encountered again
junction update_junc(junction j)
{
    if (j==ONLY_RIGHT)
    return STRAIGHT_AND_RIGHT;
    else if (j==ONLY_LEFT)
    return STRAIGHT_AND_LEFT;
    else if (j==STRAIGHT_AND_LEFT || j==STRAIGHT_AND_RIGHT) 
    return PLUS;
}

int is_present(int a) {
    
    for (int i=a-1; i>=0; i--) {
        if ((coo[i].x == coo[a].x) && (coo[i].y == coo[a].y))
        return i;
    }
    return -1;
}

//removing duplicates and optimising 
void optimise_path(int ctr) {
    for (int i=0; i<ctr; i++) {
        int p = is_present(i);
        if (p>=0) {
            coo[p].junc = update_junc(coo[p].junc);
            for (int del=i; del>p; del--) {
                coo[del].x = -9999;
                coo[del].y = -9999;
            }
        }
    }
}

typedef enum {
    RIGHT, LEFT, UTURN
} turn;

typedef struct
{
    turn t; junction j; direction d;
} path;

turn dir_change_to_turn(direction d1, direction d2) {
    if ((d1==NORTH && d2 == SOUTH) || (d1==SOUTH && d2 == NORTH) || (d1==EAST && d2 == WEST) || (d1==WEST && d2 == EAST))
    return UTURN;
    if (d1==NORTH && d2==EAST) 
    return RIGHT;
    else if (d1==NORTH && d2==WEST) 
    return LEFT;
    else if (d1==SOUTH && d2==EAST) 
    return LEFT;
    else if (d1==SOUTH && d2==WEST) 
    return RIGHT;
    else if (d1==EAST && d2==NORTH) 
    return LEFT;
    else if (d1==EAST && d2==SOUTH) 
    return RIGHT;
    else if (d1==WEST && d2==NORTH) 
    return RIGHT;
    else if (d1==WEST && d2==SOUTH) 
    return LEFT;
}

path p[100];
int store_path(int size) {
    int ctr=0;
    int temp=0;

    for (int i=0; i<size; i++) {

        if (coo[i].x ==-9999 || coo[i].junc==STRAIGHT) {
            continue;
        }

        if (coo[i].junc != STRAIGHT) {
            p[ctr].j = coo[i].junc;
            temp=i;
            while (coo[i].x == -9999) {
                i++;
            }
            if (i+1>=size)
            break;
            p[ctr].d = coo[i+1].dir;
            p[ctr].t = dir_change_to_turn(coo[temp].dir, coo[i].dir);
            ctr++;
        }
        
    }
    return ctr+1;
}


int main() 
{
   readRawData("LSA_IR.txt");
   convertToBinary(input_raw, dataCount);
   int ctr = convertLSA_xy(input_binary, coo);
   optimise_path(ctr);
   int pathsize = store_path(ctr);
   printf("START \n");
   for (int i=0; i<pathsize; i++) {
    printf("%d\t%d\t%d\n", p[i].t, p[i].j, p[i].d);
   }
   printf("END");

   return 0;
}
