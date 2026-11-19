// SPDX-License-Identifier: MIT
#include "forte/handwired/HANDWIRED_RUN_fbt.h"

#include <array>
#include <fstream>
#include <iterator>
#include <string>

using namespace forte::literals;

namespace forte::handwired {
  namespace {
    const auto cDataInputNames = std::array{"QI"_STRID, "PATH"_STRID, "IN"_STRID};
    const auto cDataOutputNames = std::array{"QO"_STRID, "STATUS"_STRID, "OUT"_STRID};
    const auto cEventInputNames = std::array{"INIT"_STRID, "REQ"_STRID};
    const auto cEventInputTypeIds = std::array{"EInit"_STRID, "Event"_STRID};
    const auto cEventOutputNames = std::array{"INITO"_STRID, "CNF"_STRID};
    const auto cEventOutputTypeIds = std::array{"EInit"_STRID, "Event"_STRID};

    const SFBInterfaceSpec cFBInterfaceSpec = {
        .mEINames = cEventInputNames,
        .mEITypeNames = cEventInputTypeIds,
        .mEONames = cEventOutputNames,
        .mEOTypeNames = cEventOutputTypeIds,
        .mDINames = cDataInputNames,
        .mDONames = cDataOutputNames,
        .mDIONames = {},
        .mSocketNames = {},
        .mPlugNames = {},
    };

    // The words of an error, as handwired-run prints them.
    std::string textOf(const hw_error &paError) {
      std::array<char, 256> text{};
      hw_error_text(&paError, text.data(), text.size());
      return text.data();
    }
  } // namespace

  DEFINE_FIRMWARE_FB(FORTE_HANDWIRED_RUN, "handwired::HANDWIRED_RUN"_STRID)

  FORTE_HANDWIRED_RUN::FORTE_HANDWIRED_RUN(const StringId paInstanceNameId, CFBContainer &paContainer) :
      CFunctionBlock(paContainer, cFBInterfaceSpec, paInstanceNameId),
      var_QI(false_BOOL),
      var_PATH(""_STRING),
      var_IN(),
      var_QO(false_BOOL),
      var_STATUS(""_STRING),
      var_OUT(),
      conn_INITO(*this, scmEventINITOID),
      conn_CNF(*this, scmEventCNFID),
      conn_QI(nullptr),
      conn_PATH(nullptr),
      conn_IN(nullptr),
      conn_QO(*this, 0, var_QO),
      conn_STATUS(*this, 1, var_STATUS),
      conn_OUT(*this, 2, var_OUT),
      mProgram() {};

  void FORTE_HANDWIRED_RUN::executeEvent(TEventID paEIID, CEventChainExecutionThread *const paECET) {
    switch (paEIID) {
      case scmEventINITID:
        if (var_QI) {
          load();
        } else {
          unload();
          var_QO = false_BOOL;
          var_STATUS = CIEC_STRING(std::string("unloaded"));
        }
        sendOutputEvent(scmEventINITOID, paECET);
        break;
      case scmEventREQID:
        if (!var_QI) {
          var_QO = false_BOOL;
          var_STATUS = CIEC_STRING(std::string("QI is FALSE"));
        } else if (!mLoaded) {
          var_QO = false_BOOL;
          var_STATUS = CIEC_STRING(std::string("no program loaded: INIT with QI = TRUE first"));
        } else {
          run();
        }
        sendOutputEvent(scmEventCNFID, paECET);
        break;
      default: break;
    }
  }

  void FORTE_HANDWIRED_RUN::load() {
    unload();
    var_QO = false_BOOL;
    const std::string path = var_PATH.getStorage();
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      var_STATUS = CIEC_STRING("can't read " + path);
      return;
    }
    mBytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    hw_error error;
    if (hw_program_parse(mBytes.data(), mBytes.size(), &mProgram, &error) != HW_OK) {
      var_STATUS = CIEC_STRING(path + ": " + textOf(error));
      mBytes.clear();
      return;
    }
    const size_t inputs = hw_program_inputs(&mProgram);
    const size_t outputs = hw_program_outputs(&mProgram);
    if (inputs > scmCells || outputs > scmCells) {
      var_STATUS = CIEC_STRING(path + ": " + std::to_string(inputs) + " input and " + std::to_string(outputs) +
                               " output cells; this block holds " + std::to_string(scmCells) + " of each");
      mBytes.clear();
      return;
    }
    mScratch.assign(hw_program_scratch(&mProgram), 0.0f);
    mLoaded = true;
    // The description's first line is the function's name.
    std::array<char, 256> description{};
    hw_program_description(&mProgram, description.data(), description.size());
    std::string name = description.data();
    name = name.substr(0, name.find('\n'));
    var_QO = true_BOOL;
    var_STATUS = CIEC_STRING(name + ": " + std::to_string(inputs) + " inputs, " + std::to_string(outputs) +
                             " outputs");
  }

  void FORTE_HANDWIRED_RUN::unload() {
    mLoaded = false;
    mBytes.clear();
    mScratch.clear();
  }

  void FORTE_HANDWIRED_RUN::run() {
    const size_t inputCount = hw_program_inputs(&mProgram);
    const size_t outputCount = hw_program_outputs(&mProgram);
    std::array<float, scmCells> inputs{};
    std::array<float, scmCells> outputs{};
    for (size_t i = 0; i < inputCount; i++) {
      inputs[i] = static_cast<TForteFloat>(var_IN[static_cast<intmax_t>(i)]);
    }
    hw_error error;
    const int code = hw_program_run(&mProgram, inputs.data(), inputCount, mScratch.data(), mScratch.size(),
                                    outputs.data(), outputCount, &error);
    if (code != HW_OK) {
      var_QO = false_BOOL;
      var_STATUS = CIEC_STRING(textOf(error));
      return;
    }
    for (size_t i = 0; i < outputCount; i++) {
      var_OUT[static_cast<intmax_t>(i)] = CIEC_REAL(outputs[i]);
    }
    var_QO = true_BOOL;
    var_STATUS = CIEC_STRING(std::string("OK"));
  }

  void FORTE_HANDWIRED_RUN::readInputData(TEventID paEIID) {
    switch (paEIID) {
      case scmEventINITID:
        readData(0, var_QI, conn_QI);
        readData(1, var_PATH, conn_PATH);
        break;
      case scmEventREQID:
        readData(0, var_QI, conn_QI);
        readData(2, var_IN, conn_IN);
        break;
      default: break;
    }
  }

  void FORTE_HANDWIRED_RUN::writeOutputData(TEventID paEIID) {
    switch (paEIID) {
      case scmEventINITOID:
        writeData(cFBInterfaceSpec.getNumDIs() + 0, var_QO, conn_QO);
        writeData(cFBInterfaceSpec.getNumDIs() + 1, var_STATUS, conn_STATUS);
        break;
      case scmEventCNFID:
        writeData(cFBInterfaceSpec.getNumDIs() + 0, var_QO, conn_QO);
        writeData(cFBInterfaceSpec.getNumDIs() + 1, var_STATUS, conn_STATUS);
        writeData(cFBInterfaceSpec.getNumDIs() + 2, var_OUT, conn_OUT);
        break;
      default: break;
    }
  }

  CIEC_ANY *FORTE_HANDWIRED_RUN::getDI(size_t paIndex) {
    switch (paIndex) {
      case 0: return &var_QI;
      case 1: return &var_PATH;
      case 2: return &var_IN;
    }
    return nullptr;
  }

  CIEC_ANY *FORTE_HANDWIRED_RUN::getDO(size_t paIndex) {
    switch (paIndex) {
      case 0: return &var_QO;
      case 1: return &var_STATUS;
      case 2: return &var_OUT;
    }
    return nullptr;
  }

  CEventConnection *FORTE_HANDWIRED_RUN::getEOConUnchecked(TPortId paIndex) {
    switch (paIndex) {
      case 0: return &conn_INITO;
      case 1: return &conn_CNF;
    }
    return nullptr;
  }

  CDataConnection **FORTE_HANDWIRED_RUN::getDIConUnchecked(TPortId paIndex) {
    switch (paIndex) {
      case 0: return &conn_QI;
      case 1: return &conn_PATH;
      case 2: return &conn_IN;
    }
    return nullptr;
  }

  CDataConnection *FORTE_HANDWIRED_RUN::getDOConUnchecked(TPortId paIndex) {
    switch (paIndex) {
      case 0: return &conn_QO;
      case 1: return &conn_STATUS;
      case 2: return &conn_OUT;
    }
    return nullptr;
  }

  void FORTE_HANDWIRED_RUN::setInitialValues() {
    var_QI = false_BOOL;
    var_PATH = ""_STRING;
    var_IN = TCells();
    var_QO = false_BOOL;
    var_STATUS = ""_STRING;
    var_OUT = TCells();
  }
} // namespace forte::handwired
