// 19M18174 Dawei Shen

#include <iostream>
#include "vtr_writer.hpp"


using namespace std;

const int nx = 41;
const int ny = 41; 
const int nt = 10000;  
const int nit = 50;    
const double c = 1.0;
const double dx = 2.0 / (nx - 1);  
const double dy = 2.0 / (ny - 1);  
const double rho = 1.0;     
const double nu = 0.1;   
const double dt = 0.001; 
const int num_threads = 128;   

__global__ void build_up_b(double *b, double *u, double *v)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx;

    if( idx > 0 && idx < nx - 1 && idy > 0 && idy < ny-1 )
    {
        b[index] =
        (rho * ( 1.0/dt *
                ((u[index+1] - u[index-1]) / 
                (2 * dx)+ (v[index+nx] - v[index-nx]) / (2 * dy)) -
                ((u[index+1] - u[index-1]) / (2*dx) * (u[index+1] - u[index-1]) / (2*dx)) -
                2 *((u[index+nx] - u[index-nx]) / (2*dy) *
                    (v[index+1]  - v[index-1])  / (2*dx)) -
                ((v[index+nx] - v[index-nx]) / (2*dy) * (v[index+nx] - v[index-nx]) / (2*dy)) ));
    }    
}

__global__ void pressure_poisson(double *p, double *pn, double *b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx;
    
    if(idx > 0 && idx < nx - 1 && idy > 0 && idy < ny - 1)
    {    
        p[index] =
        (((pn[index+1] + pn[index-1]) * dy * dy +
          (pn[index+nx] + pn[index-nx]) * dx * dx)/
          (2 * (dx * dx + dy * dy)) -
          dx * dx * dy * dy / (2 * (dx *dx + dy * dy))
          * b[index] );
    }
           
    if(idx == 0 && idy < ny )
    {
        p[index] = p[index+1]; // dp/dx = 0 at left boundary
    }
    else if(idx == nx - 1 && idy < ny )
    {
        p[index] = p[index-1]; // dp/dx = 0 at right boundary
    }    
    else if(idy == 0 && idx < nx )
    {
        p[index] = p[index+nx]; // dp/dy = 0 at bottom boundary
    }
    else if(idy == ny - 1 && idx < nx)
    {
        p[index] = 0.0; // p  = 0 at top boundary
    }
}

__global__ void cavity_flow(double *u, double *v,  double *p, double *un,  double *vn, double *b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx;

    if( idx > 0 && idx < nx - 1 && idy > 0 && idy < ny - 1 )
    {
        u[index] = (
                     un[index] - 
                     un[index] * dt / dx 
                     * (un[index] - un[index-1]) -
                     vn[index] * dt / dy * 
                     (un[index] - un[index-nx]) -
                     dt / (2 * rho * dx) * (p[index+1] - p[index-1]) +
                     nu * (dt / (dx * dx) * 
                     (un[index+1] - 2*un[index] + un[index-1]) +
                     dt / (dy * dy) * 
                     (un[index+nx] - 2*un[index] + un[index-nx])));
        
        v[index] = (
                      vn[index] - un[index] * dt / dx * 
                     (vn[index] - vn[index-1]) -
                      vn[index] * dt / dy * 
                     (vn[index] - vn[index-nx]) -
                      dt / (2 * rho * dy) * (p[index+nx] - p[index-nx]) +
                      nu * (dt / (dx * dx) * 
                     (vn[index+1] - 2*vn[index] + vn[index-1]) +
                      dt / (dy * dy) * 
                     (vn[index+nx] - 2*vn[index] + vn[index-nx])));
    }
    
    if( idx < nx )
    {
        u[0*nx+idx] = 0.0;
        u[(nx-1)*nx+idx] = 1.0;
        v[0*nx+idx] = 0.0;
        v[(nx-1)*nx+idx] = 0.0;
    }

    if( idy < ny  )
    {
        u[idy*nx+0] = 0.0;
        u[idy*nx+nx-1] = 0.0;
        v[idy*nx+0] = 0.0;
        v[idy*nx+nx-1] = 0.0;
    }
}

void output_vtr(int outputStep, double simTime, double *u, double *v, double *p)
{
    // file name & dir path setting
    char output_dir[128],filename[128],dir_path[128];
    sprintf(output_dir, ".");
    sprintf(filename, "CavityFlow");
    sprintf(dir_path, "%s/%s", output_dir, filename);

    // make buffers for cell centered data
    std::vector<double> buff_p;
    std::vector<double> buff_u, buff_v, buff_w;
    std::vector<double> x,y,z;


    const int ist = 0, ien = nx - 1 ;
    const int jst = 0, jen = ny - 1 ;

    // set coordinate
    for(int j=jst; j<=jen; j++){ y.push_back( (j - jst)*dy ); }
    for(int i=ist; i<=ien; i++){ x.push_back( (i - ist)*dx ); }

    z.push_back( 0.0 );

    for(int j = jst; j < jen; j++)
    for(int i = ist; i < ien; i++)
    {
        buff_u.push_back( u[i+j*nx] );
        buff_v.push_back( v[i+j*nx] );

        buff_w.push_back( 0.0 );

        buff_p.push_back( p[i+j*nx] );
    }
    flow::vtr_writer vtr;
    vtr.init(dir_path,filename, nx , ny , 1 , 0, nx - 1 , 0, ny - 1, 0, 0, true);
    vtr.set_coordinate(&x.front(),&y.front(),&z.front());
    vtr.push_cell_array("velocity", &buff_u.front(), &buff_v.front(), &buff_w.front());
    vtr.push_cell_array("pressure", &buff_p.front());
    vtr.set_current_step(outputStep);
    vtr.write(simTime);
}

int main(void)
{
    int size = nx * ny * sizeof(double);
    double *u, *v, *p, *un, *vn, *pn, *b;

    //Initialization
    //Distribute memory on CPU    
    u  = (double*)malloc(size);
    v  = (double*)malloc(size);
    p  = (double*)malloc(size);
    un = (double*)malloc(size);
    vn = (double*)malloc(size);
    pn = (double*)malloc(size);
    b  = (double*)malloc(size);
    
    memset(u,  0.0, size);
    memset(v,  0.0, size);
    memset(p,  0.0, size);
    memset(un, 0.0, size);
    memset(vn, 0.0, size);
    memset(pn, 0.0, size);
    memset(b,  0.0, size);

    //Distribute memory on GPU
    double *ug, *vg, *pg, *ung, *vng, *png, *bg;

    cudaMalloc((void**)&ug  , size);
    cudaMalloc((void**)&vg  , size);
    cudaMalloc((void**)&pg  , size);
    cudaMalloc((void**)&ung , size);
    cudaMalloc((void**)&vng , size);
    cudaMalloc((void**)&png , size);
    cudaMalloc((void**)&bg  , size);
    
    cudaMemcpy(ug,  u,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(vg,  v,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(pg,  p,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(ung, un, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vng, vn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(png, pn, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bg,  b,  size, cudaMemcpyHostToDevice);

    printf("Initialization is completed\n");
    
    dim3 dimBlock( num_threads, 1 );
    dim3 grid( (nx + num_threads - 1) / num_threads, ny );
        
    for(int T = 0; T < nt ; T++)
    {
        int outputStep = 0;
        double simTime = T*dt;
        build_up_b<<<grid,dimBlock>>>(bg, ug, vg);
        cudaDeviceSynchronize();

        for(int i = 0; i < nit; i++)
        {
            pressure_poisson<<<grid,dimBlock>>>(pg, png, bg);
            swap(pg,png);
        }
        cudaDeviceSynchronize();

        cavity_flow<<<grid,dimBlock>>>(ug, vg, png, ung, vng, bg);
        swap(ug,ung);
        swap(vg,vng);
        if (T % (nt-1) == 0)
        {
            cudaMemcpy(u, ung, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(v, vng, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(p, png, size, cudaMemcpyDeviceToHost);
            output_vtr(outputStep, simTime, u, v, p);
            outputStep++;
        }     
    }
    printf("Computation is completed\n");
    return 0;
}