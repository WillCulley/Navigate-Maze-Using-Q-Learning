#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <tuple>
#include <random>
#include <ctime>
#include <algorithm> 

class qLearningAlgorithm {
public:
    qLearningAlgorithm(); 
    void displayEnvironment();
    void generateEnvironment();
    void simulateEpisodes();
    std::tuple<bool, double, std::pair<int,int>> updateEnvironment(int action, std::pair<int,int> state);
    std::vector<std::pair<int,int>> stateHistory;
    int selectAction(int episode,std::pair<int,int> state);
    int getQMaxAction(std::pair<int,int> state);
    std::pair<int,int> executeAction(std::pair<int,int> state, int action);
    std::pair<int,int> determineNewState(std::pair<int,int> state, int action, double &reward);
    bool agentWithinBounds(std::pair<int,int> state);
    bool takeStep(std::pair<int,int> &state, int &action, int currentEpisode);
    void computeFinalPolicy();
    void executeFinalPolicy();
    void loadMazeFile();

private:
    // 100 x 100 environment initialization
    char environment[100][100];
    double alpha;
    double gamma;
    double epsilon;
    // Episodes represents the number of training episodes being used
    int episodes;
    // Learned policy represents the final optimal policy  
    int learnedPolicy[100][100];
    // Qmatrix represents the Q value for taking each action in a given state
    double qMatrix[100][100][4];
    // Seed random number generator
    std::mt19937 gen;
    std::uniform_int_distribution<> dis; 
    std::uniform_real_distribution<> disDouble;  
};

// Constructor
qLearningAlgorithm::qLearningAlgorithm() {
    // Initialise Q-Learning algorithm parameters
    this->alpha = 0.1;
    this->gamma = 0.9;
    this->epsilon = 1;
    this->episodes = 10000;
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            for(int k = 0; k < 4; k++){
                qMatrix[i][j][k] = 0;
            }
        }
    }
    std::random_device rd;
    gen = std::mt19937(rd());
    dis = std::uniform_int_distribution<>(0, 3); 
    disDouble = std::uniform_real_distribution<>(0.0, 1.0);
}

// Display environment 
void qLearningAlgorithm::displayEnvironment() {
    for (int i = 0; i < 100; i++) { 
        for (int j = 0; j < 100; j++) {
            // Output path in magenta
            if(this->environment[i][j] == 'X'){
               std::cout << "\033[35m" << 'X' << "\033[0m"; 
            }
            // Output goal in green
            else if(this->environment[i][j] == 'G'){
               std::cout << "\033[32m" << 'G' << "\033[0m"; 
            }
            // Output start in cyan
            else if(this->environment[i][j] == 'S'){
               std::cout << "\033[36m" << 'S' << "\033[0m"; 
            }
            // Output walls in red
            else if(this->environment[i][j] == 'W'){
               std::cout << "\033[31m" << 'W' << "\033[0m"; 
            }
            else{
                std::cout << this->environment[i][j];
            }
        }
        std::cout << std::endl;
    }
}

void qLearningAlgorithm::generateEnvironment(){
    // Initialise squares to 0
    for (int i = 0; i < 100; i++) { 
        for (int j = 0; j < 100; j++) { 
            this->environment[i][j] = '0';
        }
    }
    // Set starting square
    this->environment[0][0] = 'S';
    // Set goal
    this->environment[99][99] = 'G';
    // Randomly set 1000 walls (Note wall placement can make reaching the goal impossible)
    for(int i = 0; i < 1000; i++){
        int randomX = std::rand() % 100 + 1;
        int randomY = std::rand() % 100 + 1;
        // Regenerate random coordinates if they already have a wall, or are S or G
        char currentChar = this->environment[randomX][randomY];
        while(currentChar != '0'){
            randomX = std::rand() % 100 + 1;
            randomY = std::rand() % 100 + 1;
            currentChar = this->environment[randomX][randomY];
        }
        this->environment[randomX][randomY] = 'W';
    }
}

int qLearningAlgorithm::getQMaxAction(std::pair<int,int> state){
    // Get largest action value for current state
    double largestQValue = -INFINITY;
    int largestActionIndex = 0;
    std::vector<int> actionIndices;
    for(int i = 0; i < 4; i ++){
        double currentQValue = qMatrix[state.first][state.second][i];
        // If Q value bigger than largestQ
        if (currentQValue > largestQValue) {
            // Assign new largest Q value to largestQValue
            largestQValue = currentQValue;
            // Replace previous duplicate largest Q values with newest one
            actionIndices.clear();  
            actionIndices.push_back(i);  
        }
        else if (currentQValue == largestQValue) {
            // Add duplicate largest Q value to indicies
            actionIndices.push_back(i);
        }
    }
    // Pick random index from 0 - number of matching largest Q values
    dis = std::uniform_int_distribution<>(0, actionIndices.size()-1);
    largestActionIndex = actionIndices[dis(gen)];
    // Reset RNG range to 0-3
    dis = std::uniform_int_distribution<>(0,3);
    return largestActionIndex;
}

int qLearningAlgorithm::selectAction(int episode, std::pair<int,int> state){
    // Randomly choose first 1000 actions
    if(episode < 100){
        int action = dis(gen);
        return action;
    }
    
    double randomNumber = disDouble(gen);
    
    // Generate number to determine exploration or exploitation
    // Determine if exploring
    if(randomNumber < epsilon){
        int action = dis(gen);
        return action;
    }
    // If exploiting, select action with highest value in qMatrix for current state
    else{
        return getQMaxAction(state);
    }
}

std::pair<int,int> qLearningAlgorithm::executeAction(std::pair<int,int> state, int action){
    // Apply action to current state
    int x = state.first;
    int y = state.second;
    switch(action){
        // If moving North  
        case 0:
            x--;
        break;
        // If moving East
        case 1:
            y++;
        break;
        // If moving South
        case 2:
            x++;
        break;
        // If moving West
        case 3:
            y--;
        break;
    }
    return std::make_pair(x, y);

}

bool qLearningAlgorithm::agentWithinBounds(std::pair<int,int> newState){
    // Check bounds, indicate if out of bounds
    int x = newState.first;
    int y = newState.second;
    if(x < 0 || x >= 100 || y < 0 || y >= 100){
        //std::cout << "Out of bounds!" << std::endl;
        return false;
    }
    else{
        return true;
    }

}

std::pair<int,int> qLearningAlgorithm::determineNewState(std::pair<int,int> state, int action, double &reward){
    // Carry out action on agent, store its new state
    std::pair<int,int> newState = executeAction(state,action);
    // If agent has moved off the maze, stay in original state
    if(agentWithinBounds(newState) == false){
        // Punish out of bounds move
        reward = -5;
        return state;
    }
    // If agent within maze, go into new state
    else{
        return newState;
    }
}

std::tuple<bool, double, std::pair<int,int>> qLearningAlgorithm::updateEnvironment(int action, std::pair<int,int> state){
    // Determine reward for that action and if new state is terminal state
    double reward = 0;
    bool terminalState = false;
    // Update state
    std::pair<int,int> newState = determineNewState(state,action,reward);
    //std::cout << "New state: (" << x << ", " << y << ")" << std::endl;
    // If state is the goal
    if(environment[newState.first][newState.second] == 'G'){
        reward = 1;
        terminalState = true;
    }
    // If state is a wall
    else if (environment[newState.first][newState.second] == 'W')
    {
        reward = -1;
        terminalState = false;
    }
    
    // Reward movement that gets closer to goal
    if (((99 - state.first + 99 - state.second) > (99 - newState.first + 99 - newState.second)) && (environment[newState.first][newState.second] == 'W'))
    {
        reward = reward + 0.5;
    }
    return std::make_tuple(terminalState, reward, newState);
}

void qLearningAlgorithm::loadMazeFile(){
    std::ifstream inputFile("validMaze.txt");
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            // Extract character
            char character = inputFile.get();
            // Ignore newline character
            if(character == '\n'){
                // Decrement j to ignore read \n
                j--;
            } 
            else{
                // Else store character in environment
                this->environment[i][j] = character;
            }
        }
    }
}

bool qLearningAlgorithm::takeStep(std::pair<int,int> &state, int &action, int currentEpisode){
    // Return reward, new state, and indicate if in terminal state
    double brackets;
    auto result = updateEnvironment(action, state);
    bool newTerminalState = std::get<0>(result);
    double reward = std::get<1>(result);
    std::pair<int,int> newState = std::get<2>(result);
    // Keep history of state
    stateHistory.push_back(newState);
    // Select next action
    int nextAction = selectAction(currentEpisode,newState);
    // Update Q-values using the Q-learning rules
    brackets = reward + (gamma * qMatrix[newState.first][newState.second][getQMaxAction(newState)]) - qMatrix[state.first][state.second][action];
    qMatrix[state.first][state.second][action] = qMatrix[state.first][state.second][action] + (alpha * brackets);
    //double newQ = qMatrix[state.first][state.second][action];
    // Change old state to new state and old action to new action
    state = newState;
    action = nextAction;
    return newTerminalState;
}

void qLearningAlgorithm::simulateEpisodes(){
    int action;
    bool terminalState;
    int steps = 0;
    std::pair<int,int> state;
    loadMazeFile();
    for(int currentEpisode = 0; currentEpisode < this->episodes; currentEpisode++){
        std::cout << "Training episode: " << currentEpisode << std::endl;
        // Reset environment to initial state
        steps = 0;
        stateHistory.clear();
        state = {0,0};
        // Keep history of states
        stateHistory.push_back(state);
        // Select initial action
        action = selectAction(currentEpisode,state);
        terminalState = false;
        std::vector<int> maxQIndices;
        while(terminalState == false){
            terminalState = takeStep(state, action, currentEpisode);
            steps++;
        }
        // Decrease epsilon after many episodes
        if(currentEpisode > 100){
            epsilon = epsilon * 0.999;
    }
        std::cout << "Steps " << steps << std::endl;
    }
}

void qLearningAlgorithm::computeFinalPolicy(){
    std::pair<int,int> state;
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 100; j++){
            // Store best action per state according to our policy
            state.first = i;
            state.second = j;
            learnedPolicy[i][j] = getQMaxAction(state); 
        }
    }
}

void qLearningAlgorithm::executeFinalPolicy(){
    // Reset environment
    std::pair<int,int> state = {0,0};
    std::pair<int,int> newState;
    loadMazeFile();
    while(environment[state.first][state.second] != 'G'){
        // Carry out action in environment based off learned policy
        newState = executeAction(state, learnedPolicy[state.first][state.second]);
        // Break once goal has been found
        if(environment[newState.first][newState.second] == 'G'){
            break;
        }
        // Mark route taken by agent
        environment[newState.first][newState.second] = 'X';
        state = newState;
    }
}

int main() {
    // Instance of qLearning algorithm
    qLearningAlgorithm qLearning;
    // Generate environment (Generated maze can be impossible to 
    // navigate i.e. the walls prevent any possible route)
    // qLearning.generateEnvironment();
    // Simulate episodes
    qLearning.simulateEpisodes();
    // Use data from training to compute final policy
    qLearning.computeFinalPolicy();
    // Execute final policy
    qLearning.executeFinalPolicy();
    // Display environment
    qLearning.displayEnvironment();
    return 0;
}
