// GPU-Accelerated Particle Simulator using CUDA + OpenGL
// Author: Divya Punglia
// Date: Dec 2024 – Feb 2025

#include <GL/glut.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

const int NUM_PARTICLES = 10000;
const float DT = 0.01f; // Time step

// 2D particle with position and velocity
struct Particle {
    float2 position;
    float2 velocity;
};

// Shared particle array accessible by both CPU and GPU
__device__ __managed__ Particle particles[NUM_PARTICLES];

// CUDA kernel to update physics of particles
__global__ void updateParticles() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    Particle& p = particles[i];

    // Apply gravity
    float2 accel = make_float2(0.0f, -9.8f);
    p.velocity.x += accel.x * DT;
    p.velocity.y += accel.y * DT;

    // Update position
    p.position.x += p.velocity.x * DT;
    p.position.y += p.velocity.y * DT;

    // Bounce off bottom boundary
    if (p.position.y < -1.0f) {
        p.position.y = -1.0f;
        p.velocity.y *= -0.8f;
    }
}

// CPU function to randomly initialize particles
void initParticles() {
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles[i].position = { (rand() % 200 - 100) / 100.0f, (rand() % 200) / 100.0f };
        particles[i].velocity = { 0.0f, 0.0f };
    }
}

// OpenGL function to render all particles
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        glVertex2f(particles[i].position.x, particles[i].position.y);
    }
    glEnd();
    glutSwapBuffers();
}

// Called repeatedly to update the simulation and trigger rendering
void idle() {
    updateParticles<<<(NUM_PARTICLES + 255) / 256, 256>>>();
    cudaDeviceSynchronize();
    glutPostRedisplay();
}

// Main function
int main(int argc, char** argv) {
    initParticles();  // Initialize particle positions on CPU

    // OpenGL and GLUT setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 800);
    glutCreateWindow("CUDA Particle Simulator");

    glutDisplayFunc(display);   // Set rendering function
    glutIdleFunc(idle);         // Set update loop
    glPointSize(2.0f);          // Size of particles
    glClearColor(0, 0, 0, 1);   // Black background

    glutMainLoop();             // Start main event loop
    return 0;
}
