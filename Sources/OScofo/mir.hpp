#pragma once

#include <vector>
#include <fftw3.h>

// Performance timer - include header-only library
#include <performance_timer.h>

// Define COMPILE_OSCOFO_WITH_LOGGER to 1 to enable OScofo-specific logger features.
#ifndef COMPILE_OSCOFO_WITH_LOGGER
#define COMPILE_OSCOFO_WITH_LOGGER 0
#endif

#if COMPILE_OSCOFO_WITH_LOGGER
// HDF5 logging system - include header-only library
#include <hdf5_logger/hdf5_logger.hpp>
#endif

#include "log.hpp"
#include "states.hpp"
namespace OScofo {

#ifndef TWO_PI
#define TWO_PI (2 * M_PI)
#endif

// Macro for easy scoped timing in MIR
#define MIR_PERF_TIMER_SCOPE(timer, name) PERF_TIMER_SCOPE(timer, name)
// ╭─────────────────────────────────────╮
// │     Music Information Retrieval     │
// ╰─────────────────────────────────────╯
class MIR {
  public:
    MIR(float Sr, float WindowSize, float HopSize);
    ~MIR();

    void SetdBTreshold(double dB);
    void GetDescription(std::vector<double> &In, Description &Desc);
    void GetLoudness(std::vector<double> &In, Description &Desc);
    double GetdB();

    // Error handling
    bool HasErrors();
    std::vector<std::string> GetErrorMessage();
    void SetError(const std::string &message);
    void ClearError();

    // Performance timing
    PerformanceTimer m_PerformanceTimer;
    void PrintPerformanceTimingSummary() const;
    // HDF5 logging functions (lightweight replacement for MATLAB debugging)
    void logValue(const std::string &varName, float value);
    void logValue(const std::string &varName, double value);
    void logVector(const std::string &varName, const std::vector<float> &data);
    void logVector(const std::string &varName, const std::vector<double> &data);
    void exportLogsToHDF5(const std::string &filename = "mir_debug_data.h5");

  private:
    // Helpers
    std::vector<double> m_WindowingFunc;
    double Mtof(double Note, double Tunning);
    double Ftom(double Freq, double Tunning);
    double Freq2Bin(double freq, double n, double Sr);

    // FFT
    double *m_FFTIn;
    fftw_complex *m_FFTOut;
    fftw_plan m_FFTPlan;
    void GetFFTDescriptions(std::vector<double> &In, Description &Desc);

    // Env
    double m_dBTreshold = -50;
    void GetRMS(std::vector<double> &In, Description &Desc);
    void GetSpectralFlux(Description &Desc);

    // Audio
    float m_FftSize;
    float m_BlockSize;
    float m_HopSize;
    float m_Sr;
    double m_dB;
    std::vector<double> m_PreviousSpectralPower;

    // Time
    double m_EventTimeElapsed = 0.0; // ms

#if COMPILE_OSCOFO_WITH_LOGGER
    // HDF5 logging system
    std::unique_ptr<DataLogger> dataLogger;
#endif

    // Errors
    bool m_HasErrors = false;
    std::vector<std::string> m_Errors;
};
} // namespace OScofo
