#pragma once

#include <unordered_map>
#include <vector>

#include "states.hpp"

#ifndef COMPILE_WITH_MATLAB
#define COMPILE_WITH_MATLAB 0
#endif

#if COMPILE_WITH_MATLAB
#include "MatlabEngine.hpp"
using namespace matlab::engine;
class MatlabHelper;
#endif

// HDF5 logging system - include from main plugin sources
#include "../../../../Sources/hdf5_logger.hpp"

namespace OScofo {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TWO_PI
#define TWO_PI (2 * M_PI)
#endif

using PitchTemplateArray = std::vector<double>;

// ╭─────────────────────────────────────╮
// │     Markov Description Process      │
// ╰─────────────────────────────────────╯
class MDP {
  public:
    MDP(double Sr, double WindowSize, double HopSize);

#if COMPILE_WITH_MATLAB
  std::unique_ptr<MatlabHelper> mh;
#endif
    
    // HDF5 logging system
    std::unique_ptr<DataLogger> dataLogger;
    
    float loopCounter = 0.0f;

    // Init Functions
    void SetScoreStates(States States);
    void SetMinEntropy(double EntropyValue);

    void UpdateAudioTemplate();
    void UpdatePhaseValues();

    // Config Functions
    void SetPitchTemplateSigma(double f);
    void SetHarmonics(int i);
    void SetBPM(double Bpm);
    double GetLiveBPM();
    void ResetLiveBpm();
    void SetdBTreshold(double dB);

    // Get Functions
    int GetTunning();
    ActionVec GetEventActions(int Index);

    std::vector<MacroState> GetStates();
    MacroState GetState(int Index);
    double GetKappa();
    void AddState(MacroState state);
    void ClearStates();

    int GetStatesSize();
    int GetEvent(Description &Desc);
    double GetPitchSimilarity(double Freq);

    // Python For Research
    std::unordered_map<double, PitchTemplateArray> GetPitchTemplate();

    // Set Variables
    void SetTunning(double Tunning);
    void SetCurrentEvent(int Event);

    // Errors
    bool HasErrors();
    std::vector<std::string> GetErrorMessage();
    void SetError(const std::string &message);
    void ClearError();

  private:
    // Config
    double m_MinEntropy = 0;

    // Audio
    double m_Sr;
    double m_FFTSize;
    double m_HopSize;
    double m_Harmonics = 5;
    double m_dBTreshold = -55;
    int m_BufferSize = 1000;

    // Events
    double m_Tunning = 440;
    int m_CurrentStateIndex = -1;

    // Time
    double m_SyncStrength = 0.5;
    double m_PhaseCoupling = 0.5;
    double m_SyncStr = 0;
    double m_TimeInPrevEvent = 0;

    double m_LastTn = 0;
    double m_BlockDur = 0;
    double m_CurrentStateOnset = 0;
    int m_MaxScoreState = 0;

    int m_Tau = 0;
    double m_LastPsiN = 0;
    double m_PsiN = 0;
    double m_PsiN1 = 0;
    double m_BPM = 0;
    double m_Kappa = 1;
    double m_MaxAheadSeconds;
    double m_BeatsAhead = 1;
    double m_NormAlpha = 1;
    double m_SecondsAhead = 2;

    // Time
    double UpdatePsiN(int StateIndex);
    double InverseA2(double r);
    double ModPhases(double value);
    double CouplingFunction(double Phi, double PhiMu, double Kappa);
    double GetSojournTime(MacroState &State, int u);

    // Markov and Probabilities
    double GetTransProbability(int i, int j);
    std::vector<double> GetInitialDistribution();

    int GetMaxUForJ(MacroState &StateJ);

    // Pitch
    std::vector<MacroState> m_States;
    double m_PitchTemplateSigma = 0.5;
    double m_PitchScalingFactor = 0.5; // TODO: How should I call this?
    std::unordered_map<double, PitchTemplateArray> m_PitchTemplates;

    // Audio Observations
    void GetAudioObservations(int FirstStateIndex, int LastStateIndex, int T);
    void BuildPitchTemplate(double Freq);
    Description m_Desc;

    // Markov
    bool m_EventDetected = false;
    double GetBestEvent();
    int GetMaxJIndex(int StateIndex);
    int Inference(int CurrentState, int j, int T);
    double SemiMarkov(MacroState &StateJ, int CurrentState, int j, int T, int bufferIndex);
    double Markov(MacroState &StateJ, int CurrentState, int j, int T, int bufferIndex);

    // Errors
    bool m_HasErrors = false;
    std::vector<std::string> m_Errors;

    // HDF5 logging functions (lightweight replacement for MATLAB debugging)
    void logValue(const std::string &varName, float value);
    void logValue(const std::string &varName, double value);
    void logVector(const std::string &varName, const std::vector<float> &data);
    void logVector(const std::string &varName, const std::vector<double> &data);
    void exportLogsToHDF5(const std::string &filename = "oscofo_debug_data.h5");

  #if COMPILE_WITH_MATLAB
    /**
     * \brief Send vector to MATLAB workspace for debugging (validation builds only)
     * \param varName Variable name in MATLAB workspace
     * \param data Vector data to send
     */
    void sendDebugVectorToMatlab(const std::string &varName, const std::vector<float> &data);

    /**
     * \brief Append vector to existing MATLAB array for real-time visualization (validation builds only)
     * \param varName Variable name in MATLAB workspace to append to
     * \param data Vector data to append
     */
    void appendDebugVectorToMatlab(const std::string &varName, const std::vector<float> &data);

        /**
     * \brief Append vector to existing MATLAB array for real-time visualization (validation builds only)
     * \param varName Variable name in MATLAB workspace to append to
     * \param data Vector data to append
     */
    void appendDebugVectorToMatlabDouble(const std::string &varName, const std::vector<double> &data);
#endif


};
} // namespace OScofo

#if COMPILE_WITH_MATLAB
class MatlabHelper
{
public:
    MatlabHelper()
    {
        // Don't initialize in constructor - do it lazily when needed
        matlabEngine = nullptr;
        initializationAttempted = false;
        // ensureMatlabConnection();
    }
    ~MatlabHelper()
    {
        shutdownMatlabEngine();
    }

    // Lazy initialization - only attempt when actually needed
    bool ensureMatlabConnection()
    {
        if (matlabEngine)
            return true;

        if (initializationAttempted)
            return false; // Already tried and failed

        initializationAttempted = true;
        return initializeMatlabEngine();
    }

    // Find MatlabSession and connect to it
    bool initializeMatlabEngine()
    {
        try
        {
            // Add debug output (you can remove this later)
            std::cout << "Attempting to find MATLAB sessions..." << std::endl;

            // Find MatlabSession and connect to it
            auto matlabSessions = findMATLAB();
            std::cout << "Found " << matlabSessions.size() << " MATLAB sessions" << std::endl;

            if (!matlabSessions.empty())
            {
                std::cout << "Attempting to connect to first available session..." << std::endl;
                matlabEngine = connectMATLAB(matlabSessions[0]);
                std::cout << "Successfully connected to shared MATLAB session" << std::endl;
                return true;
            }
            else
            {
                std::cout << "No shared sessions found, attempting to start new MATLAB..." << std::endl;
                // If no shared session found, try starting a new one
                matlabEngine = startMATLAB();
                std::cout << "Successfully started new MATLAB session" << std::endl;
                return true;
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "MATLAB connection failed: " << e.what() << std::endl;
            matlabEngine = nullptr;
            return false;
        }
    };
    void shutdownMatlabEngine()
    {
        try
        {
            if (matlabEngine)
            {
                // Terminate MATLAB session
                terminateEngineClient();
                matlabEngine.reset();
            }
        }
        catch (const std::exception &e)
        {
            // Handle errors during shutdown gracefully
            matlabEngine.reset();
        }
    };
    void sendVectorToMatlabWorkspace(const std::string &varName, const std::vector<float> &data);
    void sendVectorToMatlabWorkspace(const std::string &varName, const std::vector<float> &data, const std::vector<size_t> &dims);

    // Append vector to existing MATLAB array using a MATLAB function
    void appendVectorToMatlabArray(const std::string &varName, const std::vector<float> &data);
    void appendVectorToMatlabArrayDouble(const std::string &varName, const std::vector<double> &data);
    // Check if MATLAB is available
    bool isMatlabAvailable() const { return matlabEngine != nullptr; }

private:
    std::unique_ptr<MATLABEngine> matlabEngine;
    bool initializationAttempted;
};
#endif // COMPILE_WITH_MATLAB