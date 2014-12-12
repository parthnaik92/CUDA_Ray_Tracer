#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define CRED 0
#define CGREEN 1
#define CBLUE 2

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
		getchar();
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


/* --------------- VECTORS -------------------- */

struct VECTOR3D{
	double x;
	double y;
	double z;
} ;


/* ----------------- VIEWPORT ----------------- */
struct VIEWPORT {
	int xvmin;
	int yvmin;
	int xvmax;
	int yvmax;
};


/* ------------------- PIXEL ------------------ */
struct PIXEL{
	int i;
	int j;
};


/* ---------------- SPHERE -------------------- */

struct SPHERE_INTERSECTION {
	double	lambda_in;
	double	lambda_out;
	VECTOR3D	normal;
	VECTOR3D point;
	bool	valid;
} ;

struct SPHERE {
	VECTOR3D center;
	double radius;
	double kd_rgb[3];
	double ks_rgb[3];
	double ka_rgb[3];
	double kr_rgb[3];
	double refraction_index;
	double shininess;
	bool mirror;
};


/* ------------------- RAY --------------------- */
struct RAY {
	VECTOR3D origin;
	VECTOR3D direction;
};


/* --------------- VECTOR BASIS ---------------- */
struct VEC_BASIS {
	VECTOR3D u;
	VECTOR3D v;
	VECTOR3D n;
};

__device__ void vec_sub (VECTOR3D *v1, VECTOR3D *v2, VECTOR3D *v3) {

	v1->x = v2->x - v3->x;
	v1->y = v2->y - v3->y;
	v1->z = v2->z - v3->z;
}

__device__ void vec_add (VECTOR3D *v1, VECTOR3D *v2, VECTOR3D *v3) {
	
	v1->x = v2->x + v3->x;
	v1->y = v2->y + v3->y;
	v1->z = v2->z + v3->z;
}

__device__ void vec_scale (double scale, VECTOR3D *v1, VECTOR3D *v2) {
	
	v1->x = scale * v2->x;
	v1->y = scale * v2->y;
	v1->z = scale * v2->z;
}

__device__ double dotproduct (VECTOR3D *v1, VECTOR3D *v2) {
	
	return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

__device__ VECTOR3D crossProduct(VECTOR3D *v1, VECTOR3D *v2) {
	VECTOR3D temp;
	temp.x =   ( (v1->y * v2->z) - (v1->z * v2->y) );
	temp.y = - ( (v1->x * v2->z) - (v1->z * v2->x) );
	temp.z =   ( (v1->x * v2->y) - (v1->y * v2->x) );
	return temp;
}


__device__ void normalize_vector (VECTOR3D *v) {
	
	double magnitude;
	
	// 1. calculate the magnitude (lerngth):
	magnitude = sqrt( dotproduct(v, v) );
	
	// 2. normalize the vector:
	v->x = v->x / magnitude;
	v->y = v->y / magnitude;
	v->z = v->z / magnitude;
}

__device__ void compute_ray(RAY* ray, VECTOR3D* view_point, VIEWPORT* viewport, PIXEL* pixel, VEC_BASIS* camera_frame, double distance) {
	float u, v;
	VECTOR3D v1, v2, v3, v4, dir;
	
	
	// 1. calculate u and v coordinates of the pixels on the image plane:
	u = (double)(viewport->xvmin) + (double)(pixel->i) + 0.5 ;  
	v = (double)(viewport->yvmin) + (double)(pixel->j) + 0.5 ;  
	
	// 2. calculate ray direction
	
	vec_scale(-distance, &v1, &camera_frame->n);
	vec_scale(u, &v2, &camera_frame->u);
	vec_scale(v, &v3, &camera_frame->v);
	
	ray->origin.x = view_point->x;  
	ray->origin.y = view_point->y;
	ray->origin.z = view_point->z;
	
	vec_add(&v4, &v1, &v2);
	vec_add(&dir, &v4, &v3);
	normalize_vector(&dir);
	
	ray->direction.x = dir.x;
	ray->direction.y = dir.y;
	ray->direction.z = dir.z;
}


__device__ void compute_reflected_ray(RAY* reflected_ray, RAY* incidence_ray, SPHERE_INTERSECTION* intersection) {
	
	double dp1;
	VECTOR3D scaled_normal, reflected_direction;
	
	// calculate dot-product between surface normal and the direction of the incidence ray:
	dp1 = dotproduct(&intersection->normal, &incidence_ray->direction);
	// scale surface normal by 2*dp1:
	dp1 = 2*dp1;
	vec_scale(dp1, &scaled_normal, &intersection->normal);
	
	vec_sub(&reflected_direction, &incidence_ray->direction, &scaled_normal);
	
	reflected_ray->origin=intersection->point;
	reflected_ray->direction=reflected_direction;
}

__device__ void compute_refracted_ray(RAY* refracted_ray, RAY* incidence_ray, SPHERE_INTERSECTION* intersection, SPHERE* intersection_sphere)
{
	VECTOR3D normal_normal = crossProduct(&intersection->normal, &incidence_ray->direction);

	float rotationMatrix[4][4]; 
	float inputMatrix[4]= {incidence_ray->direction.x, incidence_ray->direction.y, incidence_ray->direction.z, 1.0};
	float outputMatrix[4] = {0.0, 0.0, 0.0, 0.0};

	float u = normal_normal.x;
	float v = normal_normal.y;
	float w = normal_normal.z;

	VECTOR3D V1 = incidence_ray->direction;
	VECTOR3D V2 = intersection->normal;
	normalize_vector(&V1);
	normalize_vector(&V2);
	float angle = M_PI/2-acosf(dotproduct(&V1, &V2));
	angle = angle - asinf(intersection_sphere->refraction_index*sinf(angle));

	float L = (u*u + v * v + w * w); 
	float u2 = u * u;     
	float v2 = v * v;     
	float w2 = w * w;       
	rotationMatrix[0][0] = (u2 + (v2 + w2) * cos(angle)) / L;
	rotationMatrix[0][1] = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
	rotationMatrix[0][2] = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;
	rotationMatrix[0][3] = 0.0;
	rotationMatrix[1][0] = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
	rotationMatrix[1][1] = (v2 + (u2 + w2) * cos(angle)) / L;
	rotationMatrix[1][2] = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;
	rotationMatrix[1][3] = 0.0;
	rotationMatrix[2][0] = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
	rotationMatrix[2][1] = (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
	rotationMatrix[2][2] = (w2 + (u2 + v2) * cos(angle)) / L;
	rotationMatrix[2][3] = 0.0;
	rotationMatrix[3][0] = 0.0;
	rotationMatrix[3][1] = 0.0;
	rotationMatrix[3][2] = 0.0;
	rotationMatrix[3][3] = 1.0;

	for(int i = 0; i < 4; i++ )
	{           
		outputMatrix[i] = 0;             
		for(int k = 0; k < 4; k++)
			outputMatrix[i]+= rotationMatrix[i][k] * inputMatrix[k];
	}

	refracted_ray->origin=intersection->point;
	refracted_ray->direction.x=outputMatrix[0];
	refracted_ray->direction.y=outputMatrix[1];
	refracted_ray->direction.z=outputMatrix[2];
}


__device__ void compute_shadow_ray(RAY* ray, SPHERE_INTERSECTION* intersection, VECTOR3D* light) {

	VECTOR3D dir;
	
	// ray origin is in the intersection point
	ray->origin.x = intersection->point.x;
	ray->origin.y = intersection->point.y;
	ray->origin.z = intersection->point.z;
	
	// ray direction is from the intersection point towards the light:
	vec_sub(&dir, light, &intersection->point);
	normalize_vector(&dir);
	
	ray->direction.x = dir.x;
	ray->direction.y = dir.y;
	ray->direction.z = dir.z;
}


__device__ double blinnphong_shading(SPHERE_INTERSECTION *intersection, VECTOR3D* light, VECTOR3D* viewpoint, double kd, double ks, double ka, double p, double intensity, double amb_intensity) {
	
	double color_diffuse = 0.0; 
	double color_specular = 0.0;
	
	VECTOR3D l;
	VECTOR3D h;
	VECTOR3D v;
	
	
	// compute vector v :
	vec_sub(&v, viewpoint, &intersection->point);
	normalize_vector(&v);
	
	// compute vector l :
	vec_sub(&l, light, &intersection->point);
	normalize_vector(&l);
	
	// compute vector h:
	vec_add(&h, &v, &l);
	normalize_vector(&h);
	
	
	// compute the diffuse intensity:
	color_diffuse = kd * intensity * dotproduct(&l, &intersection->normal) ;
	if (color_diffuse < 0.0) color_diffuse = 0.0;
	
	// compute the specular intensity:
	color_specular = ks * intensity * pow (dotproduct(&h, &intersection->normal), p);
	if (color_specular < 0.0) color_specular = 0.0;
	
	return (color_diffuse + color_specular + (ka * amb_intensity));	
}
 

__device__ double shadow(double ka, double amb_intensity) {
	
	return (ka * amb_intensity);	
}


__device__ void set_rgb_array(double* rgb_array, double cred, double cgreen, double cblue) {
	rgb_array[CRED] = cred;
	rgb_array[CGREEN] = cgreen;
	rgb_array[CBLUE] = cblue;
}

__device__ bool sphere_intersection (RAY *ray, SPHERE *sphere, SPHERE_INTERSECTION* intersection) {

	double discriminant;
	double A, B, C;
	double lambda1, lambda2;
	VECTOR3D temp;
	
	A = dotproduct(&ray->direction, &ray->direction);
	
	vec_sub(&temp, &ray->origin, &sphere->center);
	B = 2 * dotproduct(&temp, &ray->direction);
	
	vec_sub(&temp, &ray->origin, &sphere->center);
	C = dotproduct(&temp, &temp) - (sphere->radius * sphere->radius);
	
	discriminant = B*B - 4*A*C;
	
	if (discriminant >= 0) {
		lambda1 = (-B + sqrt(discriminant)) / (2*A);
		lambda2 = (-B - sqrt(discriminant)) / (2*A);
		
		// is the object visible from the eye (lambda1,2>0)
		if (lambda1>=0 && lambda2>=0) {
			if (lambda1 == lambda2) {
				intersection->lambda_in = intersection->lambda_out = lambda1;
			}
			else if (lambda1 < lambda2) {
				intersection->lambda_in  = lambda1;
				intersection->lambda_out = lambda2;
			}
			else {
				intersection->lambda_in  = lambda2;
				intersection->lambda_out = lambda1;
			}
			intersection->valid = true;
			return true;
		}
		else {
			intersection->valid = false;
			return false;
		}
	}
	else {
		intersection->valid = false;
		return false;
	}

}


// Calculate normal vector in the point of intersection:
__device__ void intersection_normal(SPHERE *sphere, SPHERE_INTERSECTION* intersection, RAY* ray) {
	
	double lambda, scale;
	VECTOR3D v1, v2, point, normal;
	
	lambda = intersection->lambda_in;
	
	vec_scale(lambda, &v1, &ray->direction);
	vec_add(&point, &v1, &ray->origin);
	
	intersection->point.x = point.x;
	intersection->point.y = point.y;
	intersection->point.z = point.z;
	
	vec_sub(&v2, &point, &sphere->center);
	
	scale = 1.0 / sphere->radius;
	vec_scale(scale, &normal, &v2);
	
	normalize_vector(&normal);

	intersection->normal.x = normal.x;
	intersection->normal.y = normal.y;
	intersection->normal.z = normal.z;
	
}

__device__ void intersection_exit_normal(SPHERE *sphere, SPHERE_INTERSECTION* intersection, RAY* ray) {
	
	double lambda, scale;
	VECTOR3D v1, v2, point, normal;
	
	lambda = intersection->lambda_out;
	
	vec_scale(lambda, &v1, &ray->direction);
	vec_add(&point, &v1, &ray->origin);
	
	intersection->point.x = point.x;
	intersection->point.y = point.y;
	intersection->point.z = point.z;
	
	vec_sub(&v2, &point, &sphere->center);
	
	scale = 1.0 / sphere->radius;
	vec_scale(scale, &normal, &v2);
	
	normalize_vector(&normal);

	intersection->normal.x = normal.x;
	intersection->normal.y = normal.y;
	intersection->normal.z = normal.z;
	
}


#define NSPHERES 4
#define VIEWPLANE 400
#define WINDOW VIEWPLANE*2
#define FOCALDIST 1000
#define RADIUS 200

GLuint vbo;
void *d_vbo_buffer = NULL;

__device__ VEC_BASIS camera_frame;
__device__ VECTOR3D view_point, static_view_point;
__device__ VECTOR3D light;
__device__ SPHERE sphere[NSPHERES];
__device__ VIEWPORT viewport;

__device__ double focal_distance;
__device__ double color;
__device__ double light_intensity, ambi_light_intensity;

void Timer (int obsolete) {

	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = (VIEWPLANE<<1) * (VIEWPLANE<<1) * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

__device__ float timer=0.0f;

__global__ void animate_kernel()
{
	sphere[0].center.y=static_view_point.y+sinf(timer)*100;
	sphere[1].center.y=static_view_point.y+50+sinf(2*timer)*100;
	sphere[2].center.y=static_view_point.y+100+sinf(1.5*timer)*100;
	//view_point.x=static_view_point.x+sinf(timer)*200;
	timer+=0.02f;
}

union Color
{
	float c;
	uchar4 components;
};

//__device__ __noinline__ void calculateRefraction(double* red, double* green, double* blue, RAY ray, int intersection_object, SPHERE_INTERSECTION current_intersection, double kr, double kg, double kb, int level);

__device__ __noinline__ void calculateReflection(double* red, double* green, double* blue, RAY ray, int intersection_object, SPHERE_INTERSECTION current_intersection, double kr, double kg, double kb, int level)
{
	if (!level)
		return;
	RAY reflected_ray, shadow_ray;
	SPHERE_INTERSECTION reflected_ray_intersection, current_reflected_intersection, shadow_ray_intersection;
	compute_reflected_ray(&reflected_ray, &ray, &current_intersection);
	double reflected_theta = dotproduct(&(reflected_ray.direction), &(current_intersection.normal));
	double current_reflected_lambda = 0x7fefffffffffffff;
	double theta;
	bool bShadow=false;
	int reflected_intersection_object = -1;
	for (int l=0; l<NSPHERES; l++)
	{
		if (l!=intersection_object)
		{
			if (sphere_intersection(&reflected_ray, &sphere[l], &reflected_ray_intersection) && (reflected_theta>0.0))
			{
				intersection_normal(&sphere[l], &reflected_ray_intersection, &reflected_ray);
				if (reflected_ray_intersection.lambda_in<current_reflected_lambda)
				{
					current_reflected_lambda=reflected_ray_intersection.lambda_in;
					reflected_intersection_object=l;
					current_reflected_intersection=reflected_ray_intersection;
				}
			}
		}
	}
	if (reflected_intersection_object>=0)
	{
		compute_shadow_ray(&shadow_ray, &current_reflected_intersection, &light);
		theta = dotproduct(&(shadow_ray.direction), &(current_reflected_intersection.normal));
		for (int l=0; l<NSPHERES; l++)
		{
			if (l!=reflected_intersection_object)
			{
				if (sphere_intersection(&shadow_ray, &sphere[l], &shadow_ray_intersection) && (theta>0.0))
					bShadow=true;
			}
		}
		if (bShadow)
		{
			*red += kr*sphere[intersection_object].ks_rgb[CRED]*shadow(sphere[reflected_intersection_object].ka_rgb[CRED], ambi_light_intensity);
			*green += kg*sphere[intersection_object].ks_rgb[CGREEN]*shadow(sphere[reflected_intersection_object].ka_rgb[CGREEN], ambi_light_intensity);
			*blue += kb*sphere[intersection_object].ks_rgb[CBLUE]*shadow(sphere[reflected_intersection_object].ka_rgb[CBLUE], ambi_light_intensity);
		}
		else
		{
			*red += kr*sphere[intersection_object].ks_rgb[CRED]*blinnphong_shading(&current_reflected_intersection, &light, &view_point, sphere[reflected_intersection_object].kd_rgb[CRED], sphere[reflected_intersection_object].ks_rgb[CRED], sphere[reflected_intersection_object].ka_rgb[CRED], sphere[reflected_intersection_object].shininess,light_intensity, ambi_light_intensity);
			*green += kg*sphere[intersection_object].ks_rgb[CGREEN]*blinnphong_shading(&current_reflected_intersection, &light, &view_point, sphere[reflected_intersection_object].kd_rgb[CGREEN], sphere[reflected_intersection_object].ks_rgb[CGREEN], sphere[reflected_intersection_object].ka_rgb[CGREEN], sphere[reflected_intersection_object].shininess, light_intensity, ambi_light_intensity);
			*blue += kb*sphere[intersection_object].ks_rgb[CBLUE]*blinnphong_shading(&current_reflected_intersection, &light, &view_point,sphere[reflected_intersection_object].kd_rgb[CBLUE], sphere[reflected_intersection_object].ks_rgb[CBLUE], sphere[reflected_intersection_object].ka_rgb[CBLUE], sphere[reflected_intersection_object].shininess, light_intensity, ambi_light_intensity);
		}
		calculateReflection(red, green, blue, reflected_ray, reflected_intersection_object, current_reflected_intersection, kr*sphere[intersection_object].ks_rgb[CRED], kg*sphere[intersection_object].ks_rgb[CGREEN], kb*sphere[intersection_object].ks_rgb[CBLUE], level-1);
		//calculateRefraction(red, green, blue, reflected_ray, reflected_intersection_object, current_reflected_intersection, kr*sphere[intersection_object].kr_rgb[CRED], kg*sphere[intersection_object].kr_rgb[CGREEN], kb*sphere[intersection_object].kr_rgb[CBLUE], level-1);
	}
}

__device__ __noinline__ void calculateRefraction(double* red, double* green, double* blue, RAY ray, int intersection_object, SPHERE_INTERSECTION current_intersection, double kr, double kg, double kb, int level)
{
	if (!level)
		return;
	RAY refracted_ray, shadow_ray;
	SPHERE_INTERSECTION refracted_ray_intersection, shadow_ray_intersection;
	compute_refracted_ray(&refracted_ray, &ray, &current_intersection, &sphere[intersection_object]);
	sphere_intersection(&refracted_ray, &sphere[intersection_object], &refracted_ray_intersection);
	RAY tempRefractedRay = refracted_ray;
	intersection_exit_normal(&sphere[intersection_object], &refracted_ray_intersection, &refracted_ray);
	compute_refracted_ray(&refracted_ray, &tempRefractedRay, &refracted_ray_intersection, &sphere[intersection_object]);
	double current_refracted_lambda = 0x7fefffffffffffff;
	double theta;
	bool bShadow=false;
	int refracted_intersection_object = -1;
	SPHERE_INTERSECTION current_refracted_intersection;
	for (int l=0; l<NSPHERES; l++)
	{
		if (l!=intersection_object)
		{
			if (sphere_intersection(&refracted_ray, &sphere[l], &refracted_ray_intersection))
			{
				intersection_normal(&sphere[l], &refracted_ray_intersection, &refracted_ray);
				if (refracted_ray_intersection.lambda_in<current_refracted_lambda)
				{
					current_refracted_lambda=refracted_ray_intersection.lambda_in;
					refracted_intersection_object=l;
					current_refracted_intersection=refracted_ray_intersection;
				}
			}
		}
	}
	if (refracted_intersection_object>=0)
	{
		compute_shadow_ray(&shadow_ray, &current_refracted_intersection, &light);
		theta = dotproduct(&(shadow_ray.direction), &(current_refracted_intersection.normal));
		for (int l=0; l<NSPHERES; l++)
		{
			if (l!=refracted_intersection_object)
			{
				if (sphere_intersection(&shadow_ray, &sphere[l], &shadow_ray_intersection) && (theta>0.0))
					bShadow=true;
			}
		}
		if (bShadow)
		{
			*red += kr*sphere[intersection_object].kr_rgb[CRED]*shadow(sphere[refracted_intersection_object].ka_rgb[CRED], ambi_light_intensity);
			*green += kg*sphere[intersection_object].kr_rgb[CGREEN]*shadow(sphere[refracted_intersection_object].ka_rgb[CGREEN], ambi_light_intensity);
			*blue += kb*sphere[intersection_object].kr_rgb[CBLUE]*shadow(sphere[refracted_intersection_object].ka_rgb[CBLUE], ambi_light_intensity);
		}
		else
		{
			*red += kr*sphere[intersection_object].kr_rgb[CRED]*blinnphong_shading(&current_refracted_intersection, &light, &view_point, sphere[refracted_intersection_object].kd_rgb[CRED], sphere[refracted_intersection_object].ks_rgb[CRED], sphere[refracted_intersection_object].ka_rgb[CRED], sphere[refracted_intersection_object].shininess,light_intensity, ambi_light_intensity);
			*green += kg*sphere[intersection_object].kr_rgb[CGREEN]*blinnphong_shading(&current_refracted_intersection, &light, &view_point, sphere[refracted_intersection_object].kd_rgb[CGREEN], sphere[refracted_intersection_object].ks_rgb[CGREEN], sphere[refracted_intersection_object].ka_rgb[CGREEN], sphere[refracted_intersection_object].shininess, light_intensity, ambi_light_intensity);
			*blue += kb*sphere[intersection_object].kr_rgb[CBLUE]*blinnphong_shading(&current_refracted_intersection, &light, &view_point,sphere[refracted_intersection_object].kd_rgb[CBLUE], sphere[refracted_intersection_object].ks_rgb[CBLUE], sphere[refracted_intersection_object].ka_rgb[CBLUE], sphere[refracted_intersection_object].shininess, light_intensity, ambi_light_intensity);
		}
		//calculateRefraction(red, green, blue, refracted_ray, refracted_intersection_object, current_refracted_intersection, kr*sphere[intersection_object].kr_rgb[CRED], kg*sphere[intersection_object].kr_rgb[CGREEN], kb*sphere[intersection_object].kr_rgb[CBLUE], level-1);
		//(*calcRefr)(red, green, blue, refracted_ray, refracted_intersection_object, current_refracted_intersection, kr*sphere[intersection_object].kr_rgb[CRED], kg*sphere[intersection_object].kr_rgb[CGREEN], kb*sphere[intersection_object].kr_rgb[CBLUE], level-1);
		calculateReflection(red, green, blue, refracted_ray, refracted_intersection_object, current_refracted_intersection, kr*sphere[intersection_object].ks_rgb[CRED], kg*sphere[intersection_object].ks_rgb[CGREEN], kb*sphere[intersection_object].ks_rgb[CBLUE], level-1);
	}
}

__global__ void init_kernel()
{
	/*calcRefl=calculateReflection;
	calcRefr=calculateRefraction;*/

	// set scene:
	viewport.xvmin = -VIEWPLANE;
	viewport.yvmin = -VIEWPLANE;
	viewport.xvmax = VIEWPLANE;
	viewport.yvmax = VIEWPLANE;
	
	camera_frame.u.x = 1.0;
	camera_frame.u.y = 0.0;
	camera_frame.u.z = 0.0;
	
	camera_frame.v.x = 0.0;
	camera_frame.v.y = 1.0;
	camera_frame.v.z = 0.0;
	
	camera_frame.n.x = 0.0;
	camera_frame.n.y = 0.0;
	camera_frame.n.z = 1.0;
	
	view_point.x = (viewport.xvmax - viewport.xvmin) / 2.0 ;
	view_point.y = (viewport.yvmax - viewport.yvmin) / 2.0 ;
	view_point.z = 0.0;
	static_view_point=view_point;
	
	
	light.x = view_point.x - 1300;
	light.y = view_point.y + 1300;
	light.z = view_point.z - 300;
	
	
	ambi_light_intensity = 1.0;
	light_intensity = 1.0;
	
	focal_distance = FOCALDIST;
	
	
	sphere[0].radius = RADIUS/1.5;
	sphere[0].center.x  = view_point.x - (RADIUS+30);
	sphere[0].center.y  = view_point.y ;
	sphere[0].center.z  = view_point.z - focal_distance - (2*RADIUS+20);
	// the first sphere is blue:
	set_rgb_array(sphere[0].kd_rgb, 0.0, 0.0, 0.8);
	set_rgb_array(sphere[0].ks_rgb, 1.0, 1.0, 1.0);
	set_rgb_array(sphere[0].ka_rgb, 0.0, 0.0, 0.2);
	set_rgb_array(sphere[0].kr_rgb, 0.0, 0.0, 0.0);
	sphere[0].shininess = 100.0;
	sphere[0].refraction_index=1.52;
	sphere[0].mirror = false;
	
	sphere[1].radius = RADIUS/1.2;
	sphere[1].center.x  = view_point.x + 0;
	sphere[1].center.y  = view_point.y + 50;
	sphere[1].center.z  = view_point.z - focal_distance - (3*RADIUS+20);
	// the second sphere is green:
	set_rgb_array(sphere[1].kd_rgb, 0.0, 0.8, 0.0);
	set_rgb_array(sphere[1].ks_rgb, 0.5, 0.5, 0.5);
	set_rgb_array(sphere[1].ka_rgb, 0.0, 0.2, 0.0);
	set_rgb_array(sphere[1].kr_rgb, 0.5, 0.5, 0.5);
	sphere[1].shininess = 10.0;
	sphere[1].refraction_index=1.52;
	sphere[1].mirror = false;
	
	
	sphere[2].radius = RADIUS;
	sphere[2].center.x  = view_point.x + (2*RADIUS+30);
	sphere[2].center.y  = view_point.y + 100;
	sphere[2].center.z  = view_point.z - focal_distance - (4*RADIUS+20);
	// the third sphere is red:
	set_rgb_array(sphere[2].kd_rgb, 0.8, 0.0, 0.0);
	set_rgb_array(sphere[2].ks_rgb, 0.7, 0.7, 0.7);
	set_rgb_array(sphere[2].ka_rgb, 0.2, 0.0, 0.0);
	set_rgb_array(sphere[2].kr_rgb, 0.3, 0.3, 0.3);
	sphere[2].shininess = 100.0;
	sphere[2].refraction_index=1.52;
	sphere[2].mirror = false;
	
	
	sphere[3].radius = 100*RADIUS;
	sphere[3].center.x  = view_point.x ;
	sphere[3].center.y  = view_point.y - 100*RADIUS-130;
	sphere[3].center.z  = view_point.z - focal_distance - (4*RADIUS+20);
	// the third sphere is red:
	set_rgb_array(sphere[3].kd_rgb, 0.2, 0.2, 0.2);
	set_rgb_array(sphere[3].ks_rgb, 0.8, 0.8, 0.5);
	set_rgb_array(sphere[3].ka_rgb, 0.0, 0.0, 0.0);
	set_rgb_array(sphere[3].kr_rgb, 0.2, 0.2, 0.5);
	sphere[3].shininess = 100.0;
	sphere[3].refraction_index=1.52;
	sphere[3].mirror = true;
}

__global__ void rayTrace_kernel(float3* pos)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i>=(viewport.xvmax - viewport.xvmin) || j>(viewport.yvmax - viewport.yvmin))
		return;
	int intersection_object = -1; // none
	int reflected_intersection_object = -1; // none
	double current_lambda = 0x7fefffffffffffff; // maximum positive double
	double current_reflected_lambda = 0x7fefffffffffffff; // maximum positive double

	RAY ray, shadow_ray;
	PIXEL pixel;
	SPHERE_INTERSECTION intersection, current_intersection, shadow_ray_intersection, current_reflected_intersection;

	double red, green, blue;
	double theta, reflected_theta;

	bool bShadow = false;

	pixel.i = i;
	pixel.j = j;
			
	// 1. compute ray:
	compute_ray(&ray, &view_point, &viewport, &pixel, &camera_frame, focal_distance);
			
	// 2. check if ray hits an object:
	for (int k=0; k<NSPHERES; k++)
	{
		if (sphere_intersection(&ray, &sphere[k], &intersection))
		{
			intersection_normal(&sphere[k], &intersection, &ray);
			if (intersection.lambda_in<current_lambda)
			{
				current_lambda=intersection.lambda_in;
				intersection_object=k;
				//copy_intersection_struct(&current_intersection, &intersection);
				current_intersection=intersection;
			}
		}
	}
			
	// Compute the color of the pixel:
	if (intersection_object > -1)
	{
		compute_shadow_ray(&shadow_ray, &current_intersection, &light);
		theta = dotproduct(&(shadow_ray.direction), &(current_intersection.normal));
		for (int l=0; l<NSPHERES; l++)
		{
			if (l!=intersection_object)
			{
				if (sphere_intersection(&shadow_ray, &sphere[l], &shadow_ray_intersection) && (theta>0.0))
					bShadow=true;
			}
		}
		red=green=blue=0;

		// Reflection:
		calculateReflection(&red, &green, &blue, ray, intersection_object, current_intersection, 1.0, 1.0, 1.0, 10);

		// Refraction:
		calculateRefraction(&red, &green, &blue, ray, intersection_object, current_intersection, 1.0, 1.0, 1.0, 10);

		if (bShadow)
		{
			red += shadow(sphere[intersection_object].ka_rgb[CRED], ambi_light_intensity);
			green += shadow(sphere[intersection_object].ka_rgb[CGREEN], ambi_light_intensity);
			blue += shadow(sphere[intersection_object].ka_rgb[CBLUE], ambi_light_intensity);
		}
		else
		{
			red += blinnphong_shading(&current_intersection, &light, &view_point, sphere[intersection_object].kd_rgb[CRED], sphere[intersection_object].ks_rgb[CRED], sphere[intersection_object].ka_rgb[CRED], sphere[intersection_object].shininess, light_intensity, ambi_light_intensity);
			green += blinnphong_shading(&current_intersection, &light, &view_point, sphere[intersection_object].kd_rgb[CGREEN], sphere[intersection_object].ks_rgb[CGREEN], sphere[intersection_object].ka_rgb[CGREEN], sphere[intersection_object].shininess, light_intensity, ambi_light_intensity);
			blue += blinnphong_shading(&current_intersection, &light, &view_point, sphere[intersection_object].kd_rgb[CBLUE], sphere[intersection_object].ks_rgb[CBLUE], sphere[intersection_object].ka_rgb[CBLUE], sphere[intersection_object].shininess, light_intensity, ambi_light_intensity);
		}
		Color temp;
		if (red>1.0)
			red=1.0;
		if (green>1.0)
			green=1.0;
		if (blue>1.0)
			blue=1.0;
		temp.components = make_uchar4((unsigned char)(red*255),(unsigned char)(green*255),(unsigned char)(blue*255),1);
		pos[i*WINDOW+j] = make_float3(i, j, temp.c);
		intersection_object = -1;
		bShadow = false;
	}
	else
	{
		Color temp;
		temp.components = make_uchar4(0,0,0,1);
		pos[i*WINDOW+j] = make_float3(i, j, temp.c);
		intersection_object = -1;
		bShadow = false;
	}
	current_lambda = 0x7fefffffffffffff;
	current_reflected_lambda = 0x7fefffffffffffff;
}

Color* mat, *mat2;
#define ANTI_ALIAS_SIZE 2

__global__ void antiAlias_kernel(/*float3* pos,*/ Color* mat, Color* mat2)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i>ANTI_ALIAS_SIZE*WINDOW || j>ANTI_ALIAS_SIZE*WINDOW)
		return;
	float Kernel[3][3] = {
		{1/9.0, 1/9.0, 1/9.0},
		{1/9.0, 1/9.0, 1/9.0},
		{1/9.0, 1/9.0, 1/9.0}
	};
	double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
	for(int k = -1; k <= 1;++k)
	{
		for(int r = -1; r <=1; ++r)
		{
			sumX += Kernel[r+1][k+1]*mat[(i - r)*WINDOW*ANTI_ALIAS_SIZE+ (j - k)].components.x;
			sumY += Kernel[r+1][k+1]*mat[(i - r)*WINDOW*ANTI_ALIAS_SIZE+ (j - k)].components.y;
			sumZ += Kernel[r+1][k+1]*mat[(i - r)*WINDOW*ANTI_ALIAS_SIZE+ (j - k)].components.z;
		}                 
	}
	Color temp;
	temp.components.x=sumX;
	temp.components.y=sumY;
	temp.components.z=sumZ;
	temp.components.w=mat[i*WINDOW*ANTI_ALIAS_SIZE+j].components.w;
	mat2[i*WINDOW*ANTI_ALIAS_SIZE+j].c=temp.c;//mat[i*WINDOW*ANTI_ALIAS_SIZE+j].c;
}

__global__ void inflate_kernel(float3* pos, Color* mat)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i>WINDOW || j>WINDOW)
		return;
	for (int x=0;x<ANTI_ALIAS_SIZE;++x)
		for (int y=0;y<ANTI_ALIAS_SIZE;++y)
			mat[(ANTI_ALIAS_SIZE*i*WINDOW+x)+ANTI_ALIAS_SIZE*j+y].c=pos[i*WINDOW+j].z;
}

__global__ void deflate_kernel(float3* pos, Color* mat)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i>WINDOW || j>WINDOW)
		return;
	pos[i*WINDOW+j].z=mat[ANTI_ALIAS_SIZE*i*WINDOW+ANTI_ALIAS_SIZE*j].c;
}

void init()
{
	init_kernel<<<1,1>>>();
	cudaMalloc(&mat, (ANTI_ALIAS_SIZE*WINDOW)*(ANTI_ALIAS_SIZE*WINDOW) * sizeof(Color));
	cudaMalloc(&mat2, (ANTI_ALIAS_SIZE*WINDOW)*(ANTI_ALIAS_SIZE*WINDOW) * sizeof(Color));
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, WINDOW, 0.0, WINDOW);
}

void disp(void)
{
	animate_kernel<<<1,1>>>();
	cudaThreadSynchronize();

	float3 *dptr;
    cudaGLMapBufferObject((void**)&dptr, vbo);

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);
	
	// RAY TRACING:
	dim3 block(32, 16, 1);
	dim3 grid(WINDOW/ block.x, WINDOW / block.y, 1);
	rayTrace_kernel<<<grid,block>>>(dptr);
	HANDLE_ERROR(cudaGetLastError());
	cudaThreadSynchronize();
	inflate_kernel<<<grid,block>>>(dptr, mat);
	cudaThreadSynchronize();
	dim3 grid2(ANTI_ALIAS_SIZE*WINDOW/ block.x, ANTI_ALIAS_SIZE*WINDOW / block.y, 1);
	antiAlias_kernel<<<grid2,block>>>(/*dptr, */mat, mat2);
	cudaThreadSynchronize();
	deflate_kernel<<<grid,block>>>(dptr, mat2);
	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4,GL_UNSIGNED_BYTE,12,(GLvoid*)8);

    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WINDOW * WINDOW);
    glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	//glutPostRedisplay();
}


int main (int argc, char** argv)
{
	// init glut:
	glutInit (&argc, argv);
	// specify the display mode to be RGB and single buffering:
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position:
	glutInitWindowPosition(100,100);
	// specify the initial window size:
	glutInitWindowSize(WINDOW,WINDOW);
	// create the window and set title:
	glutCreateWindow("Basic Ray Tracer");
	// init opengl:
	init();
	// register callback function to display graphics:
	glutDisplayFunc(disp);
	glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
		exit(0);
    }
	// call Timer():
	Timer(0);
	createVBO(&vbo);
	// enter tha main loop and process events:
	glutMainLoop();
	cudaFree(mat);
	cudaFree(mat2);
	return 0;
}