
#ifndef SIMULATED_ANNEALING_HPP
#define SIMULATED_ANNEALING_HPP

#include <networkit/base/Algorithm.hpp>
#include <networkit/auxiliary/Random.hpp>

#include <cmath>

// TODO make random stuff deterministic
// TODO Better interface for initial temperature


// Simulated Annealing algorithm
template <class State, class StateTransition>
class SimulatedAnnealing : public NetworKit::Algorithm {
public:
    virtual void run() {
        std::random_device rd;
        std::mt19937 gen(Aux::Random::getSeed());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        this->round = 0;

        this->setInitialTemperature();
        do {
            this->initRound();
            //this->summarize();
            auto candidate = this->randomTransition(this->currentState);
            double oldEnergy = this->getEnergy(this->currentState);
            double candidateEnergy = this->getUpdatedEnergy(this->currentState, candidate);
            bool accepted = (dis(gen) < this->acceptanceProbability(oldEnergy, candidateEnergy, this->temperature));
            bool switchToBest = false;//(dis(gen) > this->acceptanceProbability(this->getEnergy(this->bestState), candidateEnergy, this->temperature));
            if (this->verbose) {
                std::cout << "Round " << this->round << ". Temperature: " << this->temperature << ". Candidate with energy " << candidateEnergy << ". Transition Probability: " << this->acceptanceProbability(oldEnergy,candidateEnergy, this->temperature) << ". ";
            }
            if (accepted && switchToBest && verbose) {
                std::cout << "Transition omitted. ";
            }
            if (accepted && !switchToBest) {
                if (candidateEnergy > oldEnergy) { std::cout << "Unoptimal transition. "; }
                this->transition(this->currentState, candidate);
                if (verbose)
                    std::cout << "Transition accepted. ";
                /*if (this->getEnergy(this->currentState) < this->getEnergy(this->bestState)) {
                    this->bestState = this->currentState;
                    if (verbose)
                        std::cout << "Best state set. ";
                }*/
            } else {
                if (verbose)
                    std::cout << "Transition rejected. ";
            }

            if (switchToBest) {
                this->currentState = this->bestState;
                if (verbose)
                    std::cout << "Switched to best state. \n";
            } else {
                if (verbose)
                    std::cout << "Did not switch to best state. \n";
            }


            round++;

            this->updateTemperature(oldEnergy, candidateEnergy, accepted);
        } while (!this->endIteration());
        //this->currentState = this->bestState;
        this->hasRun = true;
    }

    virtual void setInitialState(State const & s) {
        this->currentState = State(s);
        this->bestState = this->currentState;
    }

    virtual void setInitialTemperature() {
        this->temperature = std::abs(this->getEnergy(this->currentState));
    }

    virtual void summarize() {
        std::cout << "Simulated Annealing Results Summary" << "Round: " << this->round << ". Current Energy: " << this->getEnergy(this->currentState) << ". Temperature: " << this->temperature << std::endl;
    }

    virtual double getTotalValue() {
        return this->getEnergy(this->currentState);
    }

    void setVerbose(bool v) {
        this->verbose = v;
    }

    State getState() {
        return this->currentState;
    }

    int getRound() {
        return this->round;
    }



protected:
    virtual StateTransition randomTransition(State const &state)=0;
    virtual double getEnergy(State & s) = 0;
    virtual double getUpdatedEnergy(State const & s, StateTransition & t) = 0;
    virtual void transition(State & s, StateTransition & update) = 0;


    virtual bool endIteration() {
        return this->round > 400;
    }

    virtual void initRound() {}

    virtual double acceptanceProbability(double currentEnergy, double candidateEnergy, double temperature) {
        if (candidateEnergy < currentEnergy) { return 1.0; }
        return std::exp((currentEnergy - candidateEnergy) / temperature);
    }

    virtual void updateTemperature(double previousEnergy, double candidateEnergy, bool accepted) {
        this->temperature *= 0.98;
    }


    State currentState;
    int round;
    double temperature;
    State bestState;
    bool verbose = false;
};

#endif // SIMULATED_ANNEALING_HPP
