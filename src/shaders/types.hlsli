struct plane {
    float3 n;
    float d;
};
typedef plane frustum[6];

struct box3 {
    float3 min;
    float3 max;
};