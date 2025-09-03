#ifndef OSCOFO_H
#define OSCOFO_H

#include "states.hpp"
#include "score.hpp"
#include "mdp.hpp"
#include <string>
#include <vector>

namespace OScofo {

    class OScofo {
    public:
        OScofo() = default;
        bool ParseScore(std::string ScorePath);
        void SetNewAudioParameters(int sr, int fft, int hop);
        void SetError(const std::string& errorMsg);

    private:
        std::vector<MacroState> m_States;
        Score m_Score;
        MDP m_MDP;
        int m_Sr = 44100;
        int m_FFTSize = 1024;
        int m_HopSize = 512;
    };

} // namespace OScofo

#endif // OSCOFO_H

