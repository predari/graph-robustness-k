
#ifndef SIMULATED_ANNEALING_HPP
#define SIMULATED_ANNEALING_HPP

#include <networkit/base/Algorithm.hpp>

#include <cmath>

// TODO make random stuff deterministic

// Simulated Annealing algorithm
template <class State, class StateTransition>
class SimulatedAnnealing : public NetworKit::Algorithm {
public:
    virtual void run() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        this->round = 0;

        this->setInitialTemperature();
        do {
            this->initRound();
            //this->summarize();
            auto candidate = this->randomTransition(this->currentState);
            double oldEnergy = this->getEnergy(this->currentState);
            double candidateEnergy = this->getUpdatedEnergy(this->currentState, candidate);
            bool accepted = false;
            if (dis(gen) < this->acceptanceProbability(oldEnergy, candidateEnergy, this->temperature)) {
                this->transition(this->currentState, candidate);
                accepted = true;
                if (this->getEnergy(this->currentState) < this->getEnergy(this->bestState)) {
                    this->bestState = this->currentState;
                }
            }

            if (dis(gen) > this->acceptanceProbability(this->getEnergy(this->bestState), candidateEnergy, this->temperature)) {
                this->currentState = this->bestState;
            }

            round++;

            this->updateTemperature(oldEnergy, candidateEnergy, accepted);
        } while (!this->endIteration());
        this->currentState = this->bestState;
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
};

#endif // SIMULATED_ANNEALING_HPP
