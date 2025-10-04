#pragma once

#include <unordered_map>
#include <vector>

// Only include export header when building as DLL with CMake
#ifdef OSCOFO_BUILDING_DLL
    #include "oscofo_export.h"
#else
    // Define empty export macro for static builds (Projucer)
    #define OSCOFO_API
#endif

#include "states.hpp"

// Performance timer - include header-only library
#include <performance_timer.h>

// Define COMPILE_OSCOFO_WITH_LOGGER to 1 to enable OScofo-specific logger features.
#ifndef COMPILE_OSCOFO_WITH_LOGGER
#define COMPILE_OSCOFO_WITH_LOGGER 1
#endif

#if COMPILE_OSCOFO_WITH_LOGGER
// HDF5 logging system - include header-only library
#include <hdf5_logger/hdf5_logger.hpp>
#endif

namespace OScofo {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TWO_PI
#define TWO_PI (2 * M_PI)
#endif

using PitchTemplateArray = std::vector<double>;

// Macro for easy scoped timing in MDP
#define MDP_PERF_TIMER_SCOPE(timer, name) PERF_TIMER_SCOPE(timer, name)

// ╭─────────────────────────────────────╮
// │     Markov Description Process      │
// ╰─────────────────────────────────────╯
class OSCOFO_API MDP {
  public:
    MDP(double Sr, double WindowSize, double HopSize);

#if COMPILE_OSCOFO_WITH_LOGGER
    // HDF5 logging system
    std::unique_ptr<DataLogger> dataLogger;
#endif
    
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
    
    // Temporal Coherence Functions
    void SetEnableTemporalCoherence(bool enable);
    void SetTemporalCoherenceSigmaFactor(double factor);
    bool GetEnableTemporalCoherence() const;
    double GetTemporalCoherenceSigmaFactor() const;

    // Get Functions
    int GetTunning();
    ActionVec GetEventActions(int Index);

    std::vector<MacroState> GetStates();
    MacroState GetState(int Index);
    double GetKappa();
    double GetBlockDuration();

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

    // Performance timing
    void PrintPerformanceTimingSummary() const;
    void ResetPerformanceTimers();
    std::vector<PerformanceTimer::TimingResult> GetPerformanceResults() const;

  private:
    // Config
    double m_MinEntropy = 0;
    
    // Temporal Coherence (Cuvillier 2016)
    bool m_EnableTemporalCoherence = false;
    double m_TemporalCoherenceSigmaFactor = 0.1; // Sigma as fraction of m_PsiN

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

    bool m_LogsHaveBeenExported = false;

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

    // Performance timing
    PerformanceTimer m_PerformanceTimer;

    // HDF5 logging functions (lightweight replacement for MATLAB debugging)
    void logValue(const std::string &varName, float value);
    void logValue(const std::string &varName, double value);
    void logVector(const std::string &varName, const std::vector<float> &data);
    void logVector(const std::string &varName, const std::vector<double> &data);
    void exportLogsToHDF5(const std::string &filename = "oscofo_debug_data.h5");


};
} // namespace OScofo