/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

const double EPS = 0.000001;

void addNoise_(vector<Particle>& particles, double std[])
{
	default_random_engine rand;

	normal_distribution<double> noise_x(0.0, std[0]);
	normal_distribution<double> noise_y(0.0, std[1]);
	normal_distribution<double> noise_theta(0.0, std[2]);

	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		it->x += noise_x(rand);
		it->y += noise_y(rand);
		it->theta += noise_theta(rand);
	}
}

void normalizeWeight_(vector<Particle>& particles)
{
	double total_weight = 0.0;

	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		total_weight += it->weight;
	}

	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		it->weight /= total_weight;
	}
}

void resetWeight_(vector<Particle>& particles)
{
	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		it->weight = 1.0;
	}
}

double calcWeight_(const LandmarkObs& pred, const LandmarkObs& obs, double std_landmark[])
{
	return 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])
		* exp(
			-1.0 
			* (pow(pred.x - obs.x, 2) / (2.0 * pow(std_landmark[0], 2)) 
				+ pow(pred.y - obs.y, 2) / (2.0 * pow(std_landmark[1], 2)))
		);
}

double maxWeight_(vector<Particle>& particles)
{
	double weight = 0.0;

	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		if (weight < it->weight)
		{
			weight = it->weight;
		}
	}

	return weight;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 99;

	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	addNoise_(particles, std);
	normalizeWeight_(particles);

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		double theta = it->theta;

		if (fabs(yaw_rate) < EPS)
		{
			it->x += velocity * cos(theta) * delta_t;
			it->y += velocity * sin(theta) * delta_t;
		}
		else
		{
			double v_div_yr = velocity / yaw_rate;
			double yr_m_dt = yaw_rate * delta_t;
			double th_pl_yr_m_dt = theta + yr_m_dt;

			it->x += v_div_yr * (sin(th_pl_yr_m_dt) - sin(theta));
			it->y += v_div_yr * (cos(theta) - cos(th_pl_yr_m_dt));
			it->theta += yr_m_dt;
		}
	}

	addNoise_(particles, std_pos);
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	
	for (auto oit = observations.begin(); oit != observations.end(); ++oit)
	{
		double min_dist = -1;
		int id = -1;

		for (auto pit = predicted.begin(); pit != predicted.end(); ++pit)
		{
			double distance = dist(oit->x, oit->y, pit->x, pit->y);
			if (min_dist < 0 || distance < min_dist)
			{
				min_dist = distance;
				id = pit->id;
			}
		}

		oit->id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {
	
	resetWeight_(particles);

	for (auto pit = particles.begin(); pit != particles.end(); ++pit)
	{
		double px = pit->x;
		double py = pit->y;
		double ptheta = pit->theta;

		vector<LandmarkObs> predicted, obs_mapcoord;

		for (auto lit = map_landmarks.landmark_list.begin(); lit != map_landmarks.landmark_list.end(); ++lit)
		{
			if (dist(px, py, lit->x_f, lit->y_f) < sensor_range)
			{
				LandmarkObs obs;
				obs.id = lit->id_i;
				obs.x = lit->x_f;
				obs.y = lit->y_f;
				predicted.push_back(obs);
			}
		}

		for (auto oit = observations.begin(); oit != observations.end(); ++oit)
		{
			LandmarkObs obs;
			obs.id = oit->id;
			obs.x = px + oit->x * cos(ptheta) - oit->y * sin(ptheta);
			obs.y = py + oit->x * sin(ptheta) + oit->y * cos(ptheta);
			obs_mapcoord.push_back(obs);
		}

		dataAssociation(predicted, obs_mapcoord);

		for (auto oit = obs_mapcoord.begin(); oit != obs_mapcoord.end(); ++oit)
		{
			for (auto prit = predicted.begin(); prit != predicted.end(); ++prit)
			{
				if (prit->id == oit->id)
				{
					pit->weight *= calcWeight_(*prit, *oit, std_landmark);
				}
			}
		}
	}
}

void ParticleFilter::resample() {

	default_random_engine rand;
	uniform_int_distribution<int> random_int(0, num_particles - 1);
	double maxWeight = maxWeight_(particles);
	
	vector<Particle> new_particles;

	uniform_real_distribution<double> random_dbl(0.0, maxWeight);
	int index = random_int(rand);
	double beta = 0.0;

	for (auto it = particles.begin(); it != particles.end(); ++it)
	{
		beta += random_dbl(rand) * 2.0;
		double weight = particles[index].weight;
		while (beta > weight)
		{
			beta -= weight;
			index = (index + 1) % num_particles;
			weight = particles[index].weight;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
