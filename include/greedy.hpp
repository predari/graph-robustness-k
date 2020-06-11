
#include <vector>
#include <queue>
#include <limits>
#include <iostream>

#ifndef GREEDY_H
#define GREEDY_H

#include <networkit/base/Algorithm.hpp>

template <class Item>
struct ItemWrapperType {
	Item item;
	double value;
	int lastUpdated;
};

template <class Item>
bool operator<(const ItemWrapperType<Item> &left, const ItemWrapperType<Item> &right) {
	return left.value < right.value;
}

template <class Item>
class SubmodularGreedy : public NetworKit::Algorithm {
public:
	using ItemWrapper=ItemWrapperType<Item>;
	virtual void run() override;
	virtual double objectiveDifference(Item c) = 0;
	virtual void useItem(Item i) = 0;
	virtual bool checkSolution() = 0;
	virtual bool isItemAcceptable(Item c) { return true; }
	virtual void initRound() {}
	
	// To obtain results
	int getResultSize() { return round+1; }
	std::vector<Item> getResultItems() { return results; }
	double getTotalValue() { return totalValue; }
	bool isSolution() { return validSolution; }
	void summarize() {
		std::cout << "Greedy Results Summary. ";
		if (!this->hasRun) {
			std::cout << "Not executed yet!";
		}
		std::cout << "Result Size: " << this->getResultSize() << std::endl;
		for (auto e: this->getResultItems()) {
			std::cout << "(" << e.first << ", " << e.second << "), "; 
		}
		std::cout << std::endl;
		std::cout << "Marginal Gain: " << this->getTotalValue() << std::endl;
	}

protected:
	void addItems(std::vector<Item> items);
	void resetItems();


	std::priority_queue<ItemWrapper> itemQueue;

	bool validSolution = false;
	int round=0;
	std::vector<Item> results;
	double totalValue;
};


template <class Item>
void SubmodularGreedy<Item>::addItems(std::vector<Item> items) {
    for (const auto &i : items)
    {
        ItemWrapper qe {i, std::numeric_limits<double>::infinity(), -1};
        itemQueue.push(qe);
    }
}

template <class Item>
void SubmodularGreedy<Item>::resetItems() {
	this->itemQueue = std::priority_queue<ItemWrapper>();
}

template <class Item>
void SubmodularGreedy<Item>::run() {
    this->round = 0;
    this->totalValue = 0;
    this->validSolution = false;
    this->results.clear();

    bool candidatesLeft = true;

    while (candidatesLeft)
    {
		this->initRound();

        // Get top updated entry from queue
        ItemWrapper c;
        while (true) {
            if (itemQueue.empty()) {
                candidatesLeft = false;
                break;
            } else {
                c = itemQueue.top();
				itemQueue.pop();
            }

            if (this->isItemAcceptable(c.item)) {
                if (c.lastUpdated == this->round) {
                    break; // top updated entry found.
                } else {
                    c.value = this->objectiveDifference(c.item);
                    c.lastUpdated = this->round;
                    itemQueue.push(c);
                }
            } // Else: dont put it back
        }

        if (candidatesLeft) {
            this->results.push_back(c.item);
            this->totalValue += c.value;
            this->useItem(c.item);
            
            if (this->checkSolution())
            {
                this->validSolution = true;
                break;
            }
            this->round++;
        }
    }
    this->hasRun = true;
}




#endif // GREEDY_H