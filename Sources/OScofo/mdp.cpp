#include "mdp.hpp"
#include "log.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <boost/math/special_functions/bessel.hpp>
#include <numeric>

// Define math constants in case they get undefined by other headers
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if COMPILE_WITH_MATLAB
// #include "MatlabEngine.hpp"
// using namespace matlab::engine;
#endif

namespace OScofo {

/*
    // ──────────────────────────────── REFERENCES ───────────────────────────────────────

    * GONG, R.; CUVILLIER, P.; OBIN, N.; CONT, A. Real-Time Audio-to-Score Alignment of Sin-
        ging Voice Based on Melody and Lyric Information. In: Interspeech, 2015, Dresde, Germany.
        Anais… [S.l.: s.n.], 2015.

    * CONT, A. A Coupled Duration-Focused Architecture for Real-Time Music-to-Score Align-
        ment. IEEE Transactions on Pattern Analysis and Machine Intelligence, [S.l.], v.32, n.6,
        p.974–987, 2010.

    * CONT, A. Improvement of Observation Modeling for Score Following. 2004.

    * GUÉDON, Y. Hidden Hybrid Markov/Semi-Markov Chains. Computational Statistics & Data
        Analysis, [S.l.], v.49, n.3, p.663–688, 2005.

    * LARGE, E. W.; JONES, M. R. The Dynamics of Attending: How People Track Time-Varying
        Events. Psychological Review, [S.l.], v.106, n.1, p.119–159, 1999.

    * LARGE, E. W.; PALMER, C. Perceiving Temporal Regularity in Music. Cognitive Science,
        [S.l.], v.26, n.1, p.1–37, 2002.
*/

// ╭─────────────────────────────────────╮
// │Constructor and Destructor Functions │
// ╰─────────────────────────────────────╯
MDP::MDP(double Sr, double FFTSize, double HopSize) {
    m_HopSize = HopSize;
    m_FFTSize = FFTSize;
    m_Sr = Sr;

    m_SyncStrength = 0.5;
    m_PhaseCoupling = 0.5;
    m_BlockDur = (1 / m_Sr) * HopSize;
    m_TimeInPrevEvent = 0;

    SetTunning(440);

    // Initialize HDF5 logger
    dataLogger = std::make_unique<DataLogger>();
#if COMPILE_WITH_MATLAB
    mh = std::make_unique<MatlabHelper>();
#endif
}

// ╭─────────────────────────────────────╮
// │               Errors                │
// ╰─────────────────────────────────────╯
bool MDP::HasErrors() {
    return m_HasErrors;
}

// ─────────────────────────────────────
std::vector<std::string> MDP::GetErrorMessage() {
    return m_Errors;
}

// ─────────────────────────────────────
void MDP::SetError(const std::string &message) {
    printf("Error: %s.\n", message.c_str());
    m_HasErrors = true;
    m_Errors.push_back(message);
}

// ─────────────────────────────────────
void MDP::ClearError() {
    m_HasErrors = false;
    m_Errors.clear();
}

// ─────────────────────────────────────
ActionVec MDP::GetEventActions(int Index) {
    if (Index < 0 || Index >= (int)m_States.size()) {
        return ActionVec();
    }

    MacroState State = m_States[(size_t)Index];
    return State.Actions;
}

// ─────────────────────────────────────
void MDP::SetScoreStates(States ScoreStates) {
    if (ScoreStates.size() == 0) {
        SetError("ScoreStates is empty, add some events to the score");
        return;
    }

    m_States.clear();
    m_States = ScoreStates;

    for (MacroState &State : m_States) {
        State.Obs.resize(m_BufferSize + 1, 0);
        State.Forward.resize(m_BufferSize + 1, 0);
        for (AudioState &MicroState : State.SubStates) {
            MicroState.Obs.resize(m_BufferSize + 1, 0);
            MicroState.Forward.resize(m_BufferSize + 1, 0);
        }
    }

    m_CurrentStateIndex = -1;
    m_Kappa = 1;
    m_BPM = m_States[0].BPMExpected;
    m_PsiN = (double)60.0f / m_States[0].BPMExpected;
    m_PsiN1 = (double)60.0f / m_States[0].BPMExpected;
    m_LastPsiN = (double)60.0f / m_States[0].BPMExpected;
    m_BeatsAhead = m_States[0].BPMExpected / 60 * m_SecondsAhead;
    m_CurrentStateIndex = -1;
    m_SyncStr = 0;

    UpdateAudioTemplate();
    UpdatePhaseValues();
}

// ─────────────────────────────────────
void MDP::BuildPitchTemplate(double Freq) {
    // Following Gong (2015), eq 5 and 6
    const double sigmaSemitons = m_PitchTemplateSigma;
    const double sigmaLog = sigmaSemitons / 12.0;
    const double beta = 0.5;

    double rootBinFreq = round(Freq / (m_Sr / m_FFTSize));
    if (m_PitchTemplates.find(rootBinFreq) != m_PitchTemplates.end()) {
        return;
    }

    m_PitchTemplates[rootBinFreq].resize(m_FFTSize / 2, 0.0);
    for (int k = 1; k <= m_Harmonics; ++k) {
        double harmonicFreqHz = Freq * k;
        double sigmaHz = harmonicFreqHz * (std::pow(2.0, sigmaLog) - 1.0);
        double envelope = std::exp(-beta * (k - 1));
        for (size_t i = 0; i < m_FFTSize / 2; ++i) {
            double binFreq = i * (m_Sr / static_cast<double>(m_FFTSize));
            double exponent = -0.5 * std::pow((binFreq - harmonicFreqHz) / sigmaHz, 2);
            double gaussian = (1.0 / (sigmaHz * std::sqrt(2 * M_PI))) * std::exp(exponent);
            m_PitchTemplates[rootBinFreq][i] += envelope * gaussian;
        }
    }

    // Normalize template to sum to 1 (probability distribution)
    double sum = std::accumulate(m_PitchTemplates[rootBinFreq].begin(), m_PitchTemplates[rootBinFreq].end(), 0.0);
    if (sum > 0) {
        for (auto &val : m_PitchTemplates[rootBinFreq]) {
            val = (val + 1e-12) / (sum + 1e-12); // Avoid zero probabilities
        }
    }
}

// ─────────────────────────────────────
void MDP::UpdateAudioTemplate() {
    int StateSize = (int)m_States.size();
    m_PitchTemplates.clear();

    for (int h = 0; h < StateSize; h++) {
        if (m_States[h].Type == NOTE || m_States[h].Type == TRILL) {
            for (AudioState &SubState : m_States[h].SubStates) {
                if (SubState.Type == NOTE) {
                    BuildPitchTemplate(SubState.Freq);
                }
            }
        }
    }
}

// ─────────────────────────────────────
std::unordered_map<double, PitchTemplateArray> MDP::GetPitchTemplate() {
    if (m_PitchTemplates.size() == 0) {
        SetError("PitchTemplates is empty, please report this issue");
        return m_PitchTemplates;
    }
    return m_PitchTemplates;
}

// ─────────────────────────────────────
void MDP::UpdatePhaseValues() {
}

// ╭─────────────────────────────────────╮
// │          Set|Get Functions          │
// ╰─────────────────────────────────────╯
void MDP::ClearStates() {
    m_States.clear();
}
// ─────────────────────────────────────
double MDP::GetLiveBPM() {
    return m_BPM;
}

// ─────────────────────────────────────
double MDP::GetKappa() {
    return m_Kappa;
}

// ─────────────────────────────────────
void MDP::SetBPM(double BPM) {
    m_BPM = BPM;
}

// ─────────────────────────────────────
void MDP::SetdBTreshold(double dB) {
    m_dBTreshold = dB;
}

// ─────────────────────────────────────
void MDP::SetTunning(double Tunning) {
    m_Tunning = Tunning;
}

// ─────────────────────────────────────
void MDP::SetHarmonics(int Harmonics) {
    m_Harmonics = Harmonics;
}

// ─────────────────────────────────────
void MDP::SetMinEntropy(double EntropyValue) {
    m_MinEntropy = EntropyValue;
}

// ─────────────────────────────────────
int MDP::GetTunning() {
    return m_Tunning;
}

// ─────────────────────────────────────
void MDP::SetCurrentEvent(int Event) {
    m_CurrentStateIndex = Event;
    m_Tau = 0;
}

// ─────────────────────────────────────
int MDP::GetStatesSize() {
    return m_States.size();
}
// ─────────────────────────────────────
void MDP::AddState(MacroState State) {
    m_States.push_back(State);
}
// ─────────────────────────────────────
MacroState MDP::GetState(int Index) {
    return m_States[Index];
}

// ─────────────────────────────────────
void MDP::SetPitchTemplateSigma(double f) {
    m_PitchTemplateSigma = f;
}

// ╭─────────────────────────────────────╮
// │            Time Decoding            │
// ╰─────────────────────────────────────╯
double MDP::InverseA2(double SyncStrength) {
    // SyncStrength must be between 0 and 1
    if (SyncStrength < 0) {
        return 0;
    }

    // Following Large and Jones (1999, p. 157).
    if (SyncStrength > 0.95) {
        return 10.0f;
    }

    double Low = 0.0;
    double Tol = 1e-16;
    double High = std::max(SyncStrength, 10.0);
    double Mid;

    // In my tests I never reached more than 100 iterations.
    int i;
    for (i = 0; i < 1000; ++i) {
        Mid = (Low + High) / 2.0;
        double I1 = boost::math::cyl_bessel_i(1, Mid);
        double I0 = boost::math::cyl_bessel_i(0, Mid);
        double A2Mid = I1 / I0;
        if (std::fabs(A2Mid - SyncStrength) < Tol) {
            return Mid;
        } else if (A2Mid < SyncStrength) {
            Low = Mid;
        } else {
            High = Mid;
        }
    }
    LOGE() << "InverseA2 not converged after " << i << " iterations.";
    return Mid;
}

// ─────────────────────────────────────
double MDP::CouplingFunction(double Phi, double PhiMu, double Kappa) {
    // Equation 2b from Large and Palmer (2002)
    double ExpKappa = exp(Kappa);
    double PhiNDiff = Phi - PhiMu;
    double CosTerm = cos(TWO_PI * PhiNDiff);
    double SinTerm = sin(TWO_PI * PhiNDiff);
    double PhiN = (1 / (TWO_PI * ExpKappa)) * exp(Kappa * CosTerm) * SinTerm;
    return PhiN;
}

// ─────────────────────────────────────
double MDP::ModPhases(double Phase) {
    // Following Cont (2010) conventions
    Phase = std::fmod(Phase + M_PI, TWO_PI);
    if (Phase < 0) {
        Phase += TWO_PI;
    }
    return Phase - M_PI;
}

// ─────────────────────────────────────
int MDP::GetMaxJIndex(int StateIndex) {
    if (StateIndex == -1) {
        return 1;
    }

    double TimeInCurrEvt = m_States[StateIndex].Duration - (m_TimeInPrevEvent + m_BlockDur);
    double EventOnset = TimeInCurrEvt;
    int MaxJ = StateIndex + 1;
    for (size_t i = StateIndex + 1; i < m_States.size(); i++) {
        if (EventOnset > m_SecondsAhead) {
            MaxJ = i;
            break;
        } else {
            EventOnset += m_States[i].Duration;
        }
    }

    return MaxJ;
}

// ─────────────────────────────────────
/**
 * @brief UpdatePsiN - Core temporal modeling function implementing anticipatory synchronization
 *
 * This function implements the Antescofo anticipatory score following algorithm's temporal model,
 * based on Cont (2010), Large & Palmer (1999), and Large & Jones (2002). It performs:
 * 1. Period estimation (PsiN) - the current beat period estimate
 * 2. Phase coupling - synchronization between observed and expected phases
 * 3. Predictive updating - anticipatory adjustment of future event timing
 *
 * The algorithm maintains a dynamic estimate of the musical tempo (period) that adapts
 * to the performer's tempo changes while using phase-locking to maintain synchronization.
 *
 * @param StateIndex Current score position being processed
 * @return PsiN1 - Updated period estimate for next beat/event
 */
double MDP::UpdatePsiN(int StateIndex) {
    // Case 1: Still in same state - just accumulate time and frame count
    if (StateIndex == m_CurrentStateIndex) {
        m_TimeInPrevEvent += m_BlockDur; // Accumulate time spent in current state
        m_Tau += 1;                      // Increment frame counter (for Semi-Markov)
        return m_PsiN;                   // Return current period estimate unchanged
    } else {
        // Case 2: First event (initialization)
        if (StateIndex == 0) {
            // Initialize all temporal variables from score's expected BPM
            double PsiK = 60 / m_States[0].BPMExpected; // Convert BPM to period in seconds
            m_LastPsiN = PsiK;
            m_PsiN = PsiK;
            m_PsiN1 = PsiK;
            m_States[0].OnsetObserved = 0;
            m_BPM = m_States[0].BPMExpected;
            m_CurrentStateOnset = 0;
            m_LastTn = 0;
            m_TimeInPrevEvent = 0;
            m_Tau = 0;
            m_States[1].IOIHatPhiN = m_States[1].Duration; // Initial expected phase for next event
            return m_PsiN;
        } else {
            // Case 3: Transitioning to new state - update onset timing
            m_TimeInPrevEvent += m_BlockDur;
            m_LastTn = m_CurrentStateOnset;           // Store previous onset time
            m_CurrentStateOnset += m_TimeInPrevEvent; // Calculate new onset time
        }
    }
    std::vector<float> PhiNDebugVector;

    // TEMPORAL MODEL IMPLEMENTATION
    // Following Cont (2010), Large and Palmer (1999) and Large and Jones (2002)

    // Get references to relevant score states for phase calculations
    MacroState &LastState = m_States[StateIndex - 1]; // Previous event
    MacroState &CurrentState = m_States[StateIndex];  // Current event
    MacroState &NextState = m_States[StateIndex + 1]; // Next event

    // 1. INTER-ONSET INTERVAL (IOI) CALCULATION
    // Calculate observed time interval between consecutive events
    double IOISeconds = m_CurrentStateOnset - m_LastTn;
    // Extract phase information from previous calculations
    double LastPhiN = LastState.IOIPhiN;       // Previous observed phase
    double LastHatPhiN = LastState.IOIHatPhiN; // Previous expected phase
    double HatPhiN = CurrentState.IOIHatPhiN;  // Current expected phase
    // Calculate expected phase progression based on current period estimate
    double PhiNExpected = LastPhiN + ((m_CurrentStateOnset - m_LastTn) / m_PsiN);
    CurrentState.IOIHatPhiN = PhiNExpected;
    CurrentState.OnsetObserved = m_CurrentStateOnset;

    // 2. PHASE COUPLING STRENGTH ADAPTATION (Cont 2010, Large 1999)
    // Calculate phase difference between observed and expected timing
    double PhaseDiff = (IOISeconds / m_PsiN) - HatPhiN;
    // Update synchronization strength using exponential smoothing
    // This adapts the coupling strength based on recent phase consistency
    double SyncStrength = m_SyncStr - m_SyncStrength * (m_SyncStr - cos(TWO_PI * PhaseDiff));
    // Convert synchronization strength to von Mises concentration parameter
    // Higher Kappa = stronger phase coupling = more resistant to tempo changes
    double Kappa = InverseA2(SyncStrength);
    m_SyncStr = SyncStrength;
    m_Kappa = Kappa;

    // 3. PHASE UPDATE WITH COUPLING CORRECTION
    // Calculate phase coupling correction term using von Mises distribution
    double FValueUpdate = CouplingFunction(LastPhiN, LastHatPhiN, Kappa);
    // Update observed phase: natural progression + coupling correction
    double PhiN = LastPhiN + (IOISeconds / m_LastPsiN) + (m_PhaseCoupling * FValueUpdate);
    PhiN = ModPhases(PhiN); // Wrap phase to [-π, π] range
    CurrentState.PhaseObserved = PhiN;
    CurrentState.IOIPhiN = PhiN; // Store for next iteration's LastPhiN
    // 4. ANTICIPATORY PERIOD PREDICTION
    // Calculate coupling correction for next period prediction
    double FValuePrediction = CouplingFunction(PhiN, HatPhiN, Kappa);
    // Update period estimate: current period + coupling-based adjustment
    // This is the core anticipatory mechanism - predicting future tempo
    double PsiN1 = m_PsiN * (1 + m_SyncStrength * FValuePrediction);
    // 5. FUTURE EVENT TIMING PREDICTION
    // Calculate expected timing for next event based on updated period
    double Tn1 = m_CurrentStateOnset + CurrentState.Duration * PsiN1;
    double PhiN1 = ModPhases((Tn1 - m_CurrentStateOnset) / PsiN1);
    NextState.IOIHatPhiN = PhiN1;

    // Update immediate next event timing
    NextState.OnsetExpected = Tn1;
    double LastOnsetExpected = Tn1;

    // 6. LOOKAHEAD TIMING UPDATE
    // PhiNDebugVector.push_back(static_cast<float>(CurrentState.ScorePos));            // 1
    // PhiNDebugVector.push_back(static_cast<float>(m_TimeInPrevEvent));       // 2
    // PhiNDebugVector.push_back(static_cast<float>(m_LastTn));                // 3
    // PhiNDebugVector.push_back(static_cast<float>(m_CurrentStateOnset));      // 4
    // PhiNDebugVector.push_back(static_cast<float>(IOISeconds));               // 5
    // PhiNDebugVector.push_back(static_cast<float>(LastPhiN));                 // 6
    // PhiNDebugVector.push_back(static_cast<float>(LastHatPhiN));              // 7
    // PhiNDebugVector.push_back(static_cast<float>(HatPhiN));                 // 8
    // PhiNDebugVector.push_back(static_cast<float>(PhiNExpected));            // 9
    // PhiNDebugVector.push_back(static_cast<float>(PhaseDiff));               // 10
    // PhiNDebugVector.push_back(static_cast<float>(SyncStrength));            // 11
    // PhiNDebugVector.push_back(static_cast<float>(Kappa));                   // 12
    // PhiNDebugVector.push_back(static_cast<float>(m_SyncStr));               // 13
    // PhiNDebugVector.push_back(static_cast<float>(FValueUpdate));            // 14
    // PhiNDebugVector.push_back(static_cast<float>(PhiN));                   // 15
    // PhiNDebugVector.push_back(static_cast<float>(FValuePrediction));       // 16
    // PhiNDebugVector.push_back(static_cast<float>(PsiN1));                  // 17
    // PhiNDebugVector.push_back(static_cast<float>(Tn1));                    // 18
    // PhiNDebugVector.push_back(static_cast<float>(PhiN1));                  // 19

    // Propagate tempo changes to future events (up to 20 events ahead)
    // This maintains consistent timing expectations across the anticipatory window
    for (int i = m_CurrentStateIndex + 2; i < m_CurrentStateIndex + 20; i++) {
        if ((size_t)i >= m_States.size()) {
            PhiNDebugVector.push_back(-1.0f); // Indicate no further states
            continue;
        }
        MacroState &FutureState = m_States[i];
        MacroState &PreviousFutureState = m_States[(i - 1)];
        double Duration = PreviousFutureState.Duration;
        double FutureOnset = LastOnsetExpected + Duration * PsiN1;

        // PhiNDebugVector.push_back(static_cast<float>(FutureOnset));
        FutureState.OnsetExpected = FutureOnset;
        LastOnsetExpected = FutureOnset;
    }

    // Also log to HDF5 system
    // logVector("PhiNDebugVector", PhiNDebugVector);

    // Create list of headings in debug vector
    // std::vector<std::string> headings = {
    //     "TimeInPrevEvent", "LastTn", "CurrentStateOnset", "IOISeconds", "LastPhiN",
    //     "LastHatPhiN", "HatPhiN", "PhiNExpected", "PhaseDiff", "SyncStrength",
    //     "Kappa", "m_SyncStr", "FValueUpdate", "PhiN", "FValuePrediction",
    //     "PsiN1", "Tn1", "PhiN1"
    // };

    // 7. STATE VARIABLE UPDATES
    // Update global tempo tracking variables
    m_BPM = 60.0f / m_PsiN; // Convert period back to BPM
    m_LastPsiN = m_PsiN;    // Store previous period

    // Reset frame counters when transitioning to new state
    if (StateIndex != m_CurrentStateIndex) {
        m_TimeInPrevEvent = 0;
        m_Tau = 0;
    }

    return PsiN1; // Return updated period estimate for next iteration
}

// ╭─────────────────────────────────────╮
// │     Markov Description Process      │
// ╰─────────────────────────────────────╯
void MDP::GetAudioObservations(int FirstStateIndex, int LastStateIndex, int T) {
    std::unordered_map<double, double> PitchObs;

    // Create log vector
    std::vector<double> obsKLLogVector;
    obsKLLogVector.push_back((double)loopCounter);
    obsKLLogVector.push_back((double)FirstStateIndex+1);
    obsKLLogVector.push_back((double)LastStateIndex+1);
    obsKLLogVector.push_back((double)T);

    for (int j = FirstStateIndex; j <= LastStateIndex; j++) {
        if (j < 0) {
            obsKLLogVector.push_back(0.0);
            continue;
        }

        MacroState &StateJ = m_States[j];
        int BufferIndex = (T % m_BufferSize);
        if (StateJ.Type == NOTE) {
            // TODO: Need to rethink this
            double KL = 0;
            for (AudioState AudioState : StateJ.SubStates) {
                if (PitchObs.find(AudioState.Freq) != PitchObs.end()) {
                    AudioState.Obs[BufferIndex] = PitchObs[AudioState.Freq];
                    KL = PitchObs[AudioState.Freq];
                    continue;
                }
                if (AudioState.Type == NOTE) {
                    KL = GetPitchSimilarity(AudioState.Freq);
                    PitchObs[AudioState.Freq] = KL;
                    AudioState.Obs[BufferIndex] = KL;
                }
            }
            StateJ.Obs[BufferIndex] = KL;
            obsKLLogVector.push_back(KL);
        } else if (StateJ.Type == REST) {
            // StateJ.Obs[BufferIndex] = Desc.Amp;

        } else if (StateJ.Type == TRILL) {
            double bestProb = 0;
            for (AudioState AudioState : StateJ.SubStates) {
                if (PitchObs.find(AudioState.Freq) != PitchObs.end()) {
                    AudioState.Obs[BufferIndex] = PitchObs[AudioState.Freq];
                    continue;
                }
                double KL = GetPitchSimilarity(AudioState.Freq);
                if (KL > bestProb) {
                    bestProb = KL;
                }
                AudioState.Obs[BufferIndex] = KL;
            }
            StateJ.Obs[BufferIndex] = bestProb;
        }
    }
    obsKLLogVector.resize(15);
    logVector("ObsKL", obsKLLogVector);
}

// ─────────────────────────────────────
double MDP::GetPitchSimilarity(double Freq) {
    double KLDiv = 0.0;
    double RootBinFreq = round(Freq / (m_Sr / m_FFTSize));
    PitchTemplateArray PitchTemplate;

    if (m_PitchTemplates.find(RootBinFreq) != m_PitchTemplates.end()) {
        PitchTemplate = m_PitchTemplates[RootBinFreq];
    } else {
        BuildPitchTemplate(Freq);
        PitchTemplate = m_PitchTemplates[RootBinFreq];
    }

    for (size_t i = 0; i < m_FFTSize / 2; i++) {
        double P = PitchTemplate[i];
        double Q = m_Desc.NormSpectralPower[i];
        if (P > 0 && Q > 0) {
            KLDiv += P * log(P / Q);
        } else if (P == 0 && Q >= 0) {
            KLDiv += Q;
        }
    }

    // KLDiv /= (m_Desc.StdDev + 1e-9);
    double noise_robustness = 1.0 / (1.0 + m_Desc.StdDev);
    KLDiv *= noise_robustness;

    KLDiv = exp(-m_PitchScalingFactor * KLDiv);
    return KLDiv;
}

// ─────────────────────────────────────
std::vector<double> MDP::GetInitialDistribution() {
    int Size = m_MaxScoreState - m_CurrentStateIndex;
    std::vector<double> InitialProb(Size);

    double Dur = 0;
    double Sum = 0;

    for (int i = 0; i < Size; i++) {
        // If i is negative, then no note has played yet. Probability is therefore 0?
        if (m_CurrentStateIndex + i < 0) {
            InitialProb[i] = 0;
            continue;
        }
        double DurProb = exp(-1 * (Dur / m_BeatsAhead));
        InitialProb[i] = DurProb;
        Dur += m_States[m_CurrentStateIndex + i].Duration; // Accumulate duration
        Sum += DurProb;
    }

    // Normalize
    for (int i = 0; i < Size; i++) {
        InitialProb[i] /= Sum;
    }

    return InitialProb;
}

// ─────────────────────────────────────
double MDP::GetTransProbability(int i, int j) {
    // simplest markov
    if (i + 1 == j) {
        return 1;
    } else {
        return 0;
    }
}

// ─────────────────────────────────────
double MDP::GetSojournTime(MacroState &State, int u) {
    double T = m_LastTn + (m_BlockDur * u);
    double Duration = State.Duration;
    double Sojourn = std::exp(-(T - m_LastTn) / (m_PsiN1 * Duration));
    return Sojourn;
}

// ─────────────────────────────────────
int MDP::GetMaxUForJ(MacroState &StateJ) {
    double MaxU = StateJ.Duration / m_BlockDur;
    int MaxUInt = round(MaxU);
    return MaxUInt;
}

// ─────────────────────────────────────
/**
 * @brief SemiMarkov - Core Semi-Markov Forward Algorithm Implementation
 *
 * This function implements the Semi-Markov variant of the forward algorithm for score following,
 * based on the Antescofo methodology (Cont 2010). Unlike standard Markov models that assume
 * geometric duration distributions, Semi-Markov models explicitly consider state durations,
 * making them more suitable for musical score following where note durations are important.
 *
 * THEORETICAL FOUNDATION:
 * The Semi-Markov forward algorithm computes α(j,t) = P(O₁...Oₜ, Sₜ = j) where:
 * - O₁...Oₜ are observations up to time t
 * - Sₜ = j means we're in state j at time t
 * - Duration u represents how long we've been in state j
 *
 * The key difference from regular Markov models:
 * - Standard Markov: P(duration = d) = (1-p)^(d-1) * p (geometric distribution)
 * - Semi-Markov: P(duration = d) = explicit survival function (more realistic for music)
 *
 * MUSICAL RELEVANCE:
 * In musical score following, notes have explicit durations (quarter notes, half notes, etc.).
 * Semi-Markov models capture this by allowing states to have realistic duration distributions
 * rather than assuming all events are equally likely to end at any time step.
 *
 * @param StateJ Reference to the state we're computing forward probability for
 * @param CurrentState Index of the earliest state in our search window
 * @param j Index of the current state being processed (StateJ's index)
 * @param T Current time step in the forward algorithm
 * @param bufferIndex Circular buffer index for storing/retrieving probabilities
 * @return Forward probability α(j,T) for state j at time T
 */
double MDP::SemiMarkov(MacroState &StateJ, int CurrentState, int j, int T, int bufferIndex) {
    
    // BASE CASE: Initial time step (T = 0)
    // At the first time step, the forward probability is simply:
    // α(j,0) = P(O₀|j) * P(duration ≥ 1|j) * P(initial_state = j)
    if (T == 0) {
        return StateJ.Obs[bufferIndex] *     // Current observation probability P(O₀|j)
               GetSojournTime(StateJ, T + 1) * // Survival probability P(duration ≥ 1|j)
               StateJ.InitProb;               // Initial state probability P(S₀ = j)
    }
    
    // RECURSIVE CASE: T > 0
    // For Semi-Markov models, we need to consider all possible durations u that we could
    // have been in state j. The forward probability becomes:
    // α(j,t) = P(Oₜ|j) * max_u [ P(O_{t-u+1}...O_{t-1}|j) * P(duration = u|j) * max_i α(i,t-u) * P(i→j) ]
    
    double Obs = StateJ.Obs[bufferIndex]; // Current observation probability P(Oₜ|j)
    double MaxAlpha = -std::numeric_limits<double>::infinity(); // Running maximum over all durations
    
    // DURATION LOOP: Consider all possible durations u that we could have been in state j
    // We iterate from u=1 (just entered state) to min(T, max_duration_for_state_j)
    for (int u = 1; u <= std::min(T, GetMaxUForJ(StateJ)); u++) {
        
        // STEP 1: OBSERVATION LIKELIHOOD FOR DURATION u
        // If we've been in state j for duration u, we need the probability of observing
        // all observations during this period: P(O_{t-u+1}, O_{t-u+2}, ..., O_{t-1} | j)
        // We multiply individual observation probabilities (independence assumption)
        double ProbPrevObs = 1.0;
        for (int v = 1; v < u; v++) {
            int PrevIndex = (bufferIndex - v + m_BufferSize) % m_BufferSize;
            ProbPrevObs *= StateJ.Obs[PrevIndex]; // P(O_{t-v}|j)
        }
        
        // STEP 2: DURATION MODEL (SURVIVAL FUNCTION)
        // Get the probability of staying in state j for exactly duration u
        // This is the key difference from standard Markov models
        double Sur = GetSojournTime(StateJ, u); // P(duration = u | j)
        
        // STEP 3: TRANSITION PROBABILITIES
        // Consider all possible previous states i that could have transitioned to state j
        // at time (T-u). We need: max_i [ α(i, T-u) * P(i → j) ]
        double MaxTrans = -std::numeric_limits<double>::infinity();
        
        // TODO: At the moment, why loop this if GetTransProbability is only 1 when i+1==j
        for (int i = CurrentState; i <= j; i++) {
            if (i < 0) continue; // Skip invalid state indices
            
            MacroState &StateI = m_States[i];
            int PrevIndex = (T - u) % m_BufferSize; // Buffer index for time T-u
            
            double distanceScaleFactor = 0.1;
            if (i != j) {
                // CASE A: Transition from different state i to state j
                // Probability = α(i, T-u) * P(i → j)
                double TransProb = GetTransProbability(i, j) * StateI.Forward[PrevIndex];
                // Scale this probability by distance of i to CurrentState
                int distance = i - CurrentState;
                TransProb *= std::exp(-distance * distanceScaleFactor); // Example scaling factor
                MaxTrans = std::max(MaxTrans, TransProb);
            } else {
                // CASE B: Self-transition (staying in same state j)
                // This represents the case where we were already in state j at time T-u
                // and continued to stay in it for duration u
                
                // NOTE: The commented code below would handle hierarchical states
                // where a MacroState contains multiple AudioStates that could use
                // different Markov models (some Semi-Markov, some regular Markov)
                
                // for (AudioState &SubState : StateJ.SubStates) {
                //     if (SubState.Markov == MARKOV) {
                //         int StateSize = StateJ.SubStates.size();
                //         // SubState.Forward[bufferIndex] = Markov(SubState, CurrentState, j, T, bufferIndex);
                //     }
                // }
                
                // For now, we use the previous forward probability directly
                double TransProb = StateJ.Forward[PrevIndex];
                int distance = i - CurrentState;
                TransProb *= std::exp(-distance * distanceScaleFactor); // Example scaling factor
                MaxTrans = std::max(MaxTrans, TransProb);
            }
        }
        
        // STEP 4: COMBINE ALL COMPONENTS FOR THIS DURATION
        // The contribution from duration u is:
        // P(O_{t-u+1}...O_{t-1}|j) * P(duration = u|j) * max_i[α(i,t-u) * P(i→j)]
        double MaxResult = ProbPrevObs * Sur * MaxTrans;
        
        // Keep track of the maximum over all possible durations
        MaxAlpha = std::max(MaxAlpha, MaxResult);
    }
    
    // FINAL RESULT: Multiply by current observation and return
    // α(j,t) = P(Oₜ|j) * max_u [ ... ]
    return Obs * MaxAlpha;
}

// ─────────────────────────────────────
double MDP::Markov(MacroState &StateJ, int CurrentState, int j, int T, int bufferIndex) {
    if (T == 0) {
        return StateJ.Obs[bufferIndex] * StateJ.InitProb;
    } else {
        double Obs = StateJ.Obs[bufferIndex];
        double MaxAlpha = -std::numeric_limits<double>::infinity();
        for (int i = CurrentState; i <= j; i++) {
            if (i >= 0) {
                int prevIndex = (bufferIndex - 1 + m_BufferSize) % m_BufferSize;
                double Value = GetTransProbability(i, j) * m_States[i].Forward[prevIndex];
                MaxAlpha = std::max(MaxAlpha, Value);
            }
        }
        return Obs * MaxAlpha;
    }
}

// ─────────────────────────────────────
double CalculateEntropy(const std::vector<double> &probs) {
    double entropy = 0.0;
    for (double prob : probs) {
        if (prob > 0) { // Avoid log(0) which is undefined
            entropy -= prob * log(prob);
        }
    }
    return entropy;
}

// ─────────────────────────────────────
int MDP::Inference(int CurrentState, int MaxState, int T) {
    double MaxValue = -std::numeric_limits<double>::infinity();
    int BestState = CurrentState;
    int bufferIndex = T % m_BufferSize;
    std::vector<double> preNormalizedForward;
    // Loop over current and Max (e.g. 5x)
    for (int j = CurrentState; j <= MaxState; j++) {
        if ((j < 0) || ((size_t)j >= m_States.size()))
            continue;
        MacroState &StateJ = m_States[j];
        // for (AudioState &SubState : StateJ.SubStates) {
        // if (SubState.Markov == MARKOV) {
        //     int StateSize = StateJ.SubStates.size();
        //
        //     // SubState.Forward[bufferIndex] = Markov(SubState, CurrentState, j, T, bufferIndex);
        // }
        // }
        // Update single value in Forward, in stateJ.
        if (StateJ.Markov == SEMIMARKOV) {
            StateJ.Forward[bufferIndex] = SemiMarkov(StateJ, CurrentState, j, T, bufferIndex);
        } else if (StateJ.Markov == MARKOV) {
            StateJ.Forward[bufferIndex] = Markov(StateJ, CurrentState, j, T, bufferIndex);
        }
        preNormalizedForward.push_back(StateJ.Forward[bufferIndex]);
    }

    // Sum
    double SumForward = 0;
    for (int j = CurrentState; j <= MaxState; j++) {
        if ((j < 0) || ((size_t)j >= m_States.size()))
            continue;
        SumForward += m_States[j].Forward[bufferIndex];
    }

    // Normalization
    std::vector<double> Probs;
    std::vector<double> AllProbs;
    for (int j = CurrentState; j <= MaxState; j++) {
        if ((j < 0) || ((size_t)j >= m_States.size()))
            continue;

        MacroState &StateJ = m_States[j];
        // Why no normalisation when T == 0?
        if (T != 0) {
            double Forward = StateJ.Forward[bufferIndex];
            StateJ.Forward[bufferIndex] = Forward / SumForward;
            AllProbs.push_back(StateJ.Forward[bufferIndex]);
        } else {
            AllProbs.push_back(StateJ.Forward[bufferIndex]/SumForward);
        }
        if (StateJ.Forward[bufferIndex] > MaxValue) {
            MaxValue = StateJ.Forward[bufferIndex];
            BestState = j;
        }
        Probs.push_back(StateJ.Forward[bufferIndex]);
    }
    // Make AllProbs a copy of Probs, but padded to be at least 10 long

    // TODO: Implement Cuvillier (2016)
    double Entropy = CalculateEntropy(AllProbs);
    double maxEntropy = log(AllProbs.size());
    double Conf = 1.0 - (Entropy / maxEntropy);
    // printf("config: %d, %f, %f\n", BestState, entropy, confidence);
    AllProbs.resize(15, 0.0);
    preNormalizedForward.resize(15, 0.0);
    // Also log to HDF5 system
    logVector("StateJForwardNorm", AllProbs);
    logVector("preNormalizedForward", preNormalizedForward);
    logValue("Entropy", static_cast<float>(Entropy));
    logValue("MaxEntropy", static_cast<float>(maxEntropy));
    logValue("Confidence", static_cast<float>(Conf));

    if (BestState > 0) {
        if (m_MinEntropy > 0) {
            if (Conf > m_MinEntropy) {
                return BestState;
            } else {
                return CurrentState;
            }
        }
    }
    return BestState;
}

// ─────────────────────────────────────
int MDP::GetEvent(Description &Desc) {
    loopCounter += 1.0f;

    m_Desc = Desc;
    m_MaxScoreState = GetMaxJIndex(m_CurrentStateIndex);
    logVector("NormSpectralPower", m_Desc.NormSpectralPower);
    GetAudioObservations(m_CurrentStateIndex - 1, m_MaxScoreState, m_Tau);

    // If we are in silence, and not -1, return current score position.
    // If last score position, return last score position.
    if (Desc.Silence || (size_t)m_CurrentStateIndex == m_States.size()) {
        if (m_CurrentStateIndex == -1) {
            logValue("CurrentStateIndex", static_cast<float>(m_CurrentStateIndex)); // Also log to HDF5
            logValue("ScorePos", 0.0f); // Also log to HDF5
            return 0;
        }
        return m_States[m_CurrentStateIndex].ScorePos;
    }

    if (m_Tau == 0) {
        std::vector<double> InitialProb = GetInitialDistribution();
        for (int j = m_CurrentStateIndex; j < m_MaxScoreState; j++) {
            if (j < 0) {
                continue;
            }
            MacroState &StateJ = m_States[j];
            StateJ.InitProb = InitialProb[j - m_CurrentStateIndex];
        }
    }

    int StateIndex = Inference(m_CurrentStateIndex, m_MaxScoreState, m_Tau);
    if (StateIndex == -1) {
        logValue("CurrentStateIndex", static_cast<float>(m_CurrentStateIndex)); // Also log to HDF5
        logValue("ScorePos", 0.0f); // Also log to HDF5
        return 0;
    }
    m_PsiN = UpdatePsiN(StateIndex); // Time Update

    if (m_CurrentStateIndex == 47 && m_CurrentStateIndex > 1) {
        exportLogsToHDF5("MDP_Logs.h5");
    }
    // Return Score Position
    if (m_CurrentStateIndex == StateIndex) {
        m_CurrentStateIndex = StateIndex;
        logValue("CurrentStateIndex", static_cast<float>(m_CurrentStateIndex)); // Also log to HDF5
        logValue("ScorePos", static_cast<float>(m_States[StateIndex].ScorePos)); // Also log to HDF5
        return m_States[StateIndex].ScorePos;
    } else {
        m_CurrentStateIndex = StateIndex;

        // Config
        // m_MinEntropy = m_States[StateIndex].Entropy;
        // m_SyncStrength = m_States[StateIndex].SyncStrength;
        // m_PhaseCoupling = m_States[StateIndex].PhaseCoupling;

        logValue("CurrentStateIndex", static_cast<float>(m_CurrentStateIndex)); // Also log to HDF5
        logValue("ScorePos", static_cast<float>(m_States[StateIndex].ScorePos)); // Also log to HDF5
        return m_States[StateIndex].ScorePos;
    }
}

// ============================================================================
// HDF5 Logging Functions (Lightweight MATLAB replacement)
// ============================================================================

void MDP::logValue(const std::string &varName, float value) {
    if (dataLogger) {
        dataLogger->logValue(varName, value);
    }
}

void MDP::logValue(const std::string &varName, double value) {
    if (dataLogger) {
        dataLogger->logValue(varName, value);
    }
}

void MDP::logVector(const std::string &varName, const std::vector<float> &data) {
    if (dataLogger) {
        dataLogger->logVector(varName, data);
    }
}

void MDP::logVector(const std::string &varName, const std::vector<double> &data) {
    if (dataLogger) {
        dataLogger->logVector(varName, data);
    }
}

void MDP::exportLogsToHDF5(const std::string &filename) {
    if (dataLogger) {
        dataLogger->exportToHDF5(filename);
    }
}

// ============================================================================
// MATLAB Functions (Legacy support)
// ============================================================================

#if COMPILE_WITH_MATLAB
void MDP::sendDebugVectorToMatlab(const std::string &varName, const std::vector<float> &data) {
    if (mh && mh->isMatlabAvailable()) {
        mh->sendVectorToMatlabWorkspace(varName, data);
    }
}

void MDP::appendDebugVectorToMatlab(const std::string &varName, const std::vector<float> &data) {
    if (mh && mh->isMatlabAvailable()) {
        // mh->appendVectorToMatlabArray(varName, data);
    }
}

void MDP::appendDebugVectorToMatlabDouble(const std::string &varName, const std::vector<double> &data) {
    if (mh && mh->isMatlabAvailable()) {
        // mh->appendVectorToMatlabArrayDouble(varName, data);
    }
}
#endif // COMPILE_WITH_MATLAB

} // namespace OScofo

#if COMPILE_WITH_MATLAB
// ============================================================================
// MatlabHelper Implementation
// ============================================================================

void MatlabHelper::sendVectorToMatlabWorkspace(const std::string &varName, const std::vector<float> &data) {
    auto vectorLength = data.size();
    sendVectorToMatlabWorkspace(varName, data, {1, vectorLength});
}

void MatlabHelper::sendVectorToMatlabWorkspace(const std::string &varName, const std::vector<float> &data, const std::vector<size_t> &dims) {
    try {
        // Convert std::vector<float> to MATLAB array
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> matlabArray = factory.createArray(dims, data.begin(), data.end());

        // Send to MATLAB workspace
        matlabEngine->setVariable(varName, matlabArray);
    } catch (const std::exception &e) {
        std::cout << "Error sending vector to MATLAB: " << e.what() << std::endl;
    }
}

void MatlabHelper::appendVectorToMatlabArray(const std::string &varName, const std::vector<float> &data) {
    try {
        // Convert std::vector<float> to MATLAB array
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> matlabArray = factory.createArray({1, data.size()}, data.begin(), data.end());

        // Send the new data to a temporary variable
        matlabEngine->setVariable("tempNewData", matlabArray);

        // Call MATLAB function to append data to existing array
        std::string matlabCommand = "appendToArray('" + varName + "', tempNewData);";
        matlabEngine->eval(matlab::engine::convertUTF8StringToUTF16String(matlabCommand));
    } catch (const std::exception &e) {
        std::cout << "Error appending vector to MATLAB array: " << e.what() << std::endl;
    }
}

void MatlabHelper::appendVectorToMatlabArrayDouble(const std::string &varName, const std::vector<double> &data) {
    try {
        // Convert std::vector<double> to MATLAB array
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<double> matlabArray = factory.createArray({1, data.size()}, data.begin(), data.end());

        // Send the new data to a temporary variable
        matlabEngine->setVariable("tempNewData", matlabArray);

        // Call MATLAB function to append data to existing array
        std::string matlabCommand = "appendToArray('" + varName + "', tempNewData);";
        matlabEngine->eval(matlab::engine::convertUTF8StringToUTF16String(matlabCommand));
    } catch (const std::exception &e) {
        std::cout << "Error appending vector to MATLAB array: " << e.what() << std::endl;
    }
}

#endif // COMPILE_WITH_MATLAB