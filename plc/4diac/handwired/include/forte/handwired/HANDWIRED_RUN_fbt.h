// SPDX-License-Identifier: MIT
//
// HANDWIRED_RUN: a 4diac FORTE function block that runs a program file exported by handwired-rs.
//
// INIT with QI = TRUE reads the file at PATH and checks it (checksum, format, every index); with QI = FALSE it
// unloads it. REQ runs the program on the first cells of IN, as many as the program has inputs, and writes the
// first cells of OUT. A run does the same operations in the same order whatever its inputs. Inputs outside the
// ranges, grids or types the program's function declared are refused: QO is FALSE, STATUS says why, and OUT keeps
// its previous values, so logic downstream must check QO.
#pragma once

#include <cstdint>
#include <vector>

#include "forte/funcbloc.h"
#include "forte/datatypes/forte_bool.h"
#include "forte/datatypes/forte_real.h"
#include "forte/datatypes/forte_string.h"
#include "forte/datatypes/forte_array_fixed.h"

#include "handwired_run.h"

namespace forte::handwired {
  class FORTE_HANDWIRED_RUN : public CFunctionBlock {
      DECLARE_FIRMWARE_FB(FORTE_HANDWIRED_RUN)

    public:
      // The cells of IN and OUT; a program with more inputs or outputs is refused at INIT.
      static constexpr size_t scmCells = 16;
      using TCells = CIEC_ARRAY_FIXED<CIEC_REAL, 0, scmCells - 1>;

    private:
      static const TEventID scmEventINITID = 0;
      static const TEventID scmEventREQID = 1;
      static const TEventID scmEventINITOID = 0;
      static const TEventID scmEventCNFID = 1;

      void executeEvent(TEventID paEIID, CEventChainExecutionThread *const paECET) override;
      void readInputData(TEventID paEIID) override;
      void writeOutputData(TEventID paEIID) override;

      void load();
      void unload();
      void run();

    public:
      FORTE_HANDWIRED_RUN(const StringId paInstanceNameId, CFBContainer &paContainer);

      CIEC_BOOL var_QI;
      CIEC_STRING var_PATH;
      TCells var_IN;
      CIEC_BOOL var_QO;
      CIEC_STRING var_STATUS;
      TCells var_OUT;

      CEventConnection conn_INITO;
      CEventConnection conn_CNF;
      CDataConnection *conn_QI;
      CDataConnection *conn_PATH;
      CDataConnection *conn_IN;
      COutDataConnection<CIEC_BOOL> conn_QO;
      COutDataConnection<CIEC_STRING> conn_STATUS;
      COutDataConnection<TCells> conn_OUT;

      CIEC_ANY *getDI(size_t) override;
      CIEC_ANY *getDO(size_t) override;
      CEventConnection *getEOConUnchecked(TPortId) override;
      CDataConnection **getDIConUnchecked(TPortId) override;
      CDataConnection *getDOConUnchecked(TPortId) override;

    protected:
      void setInitialValues() override;

    private:
      // The file's bytes, which the parsed program points into, and the scratch memory a run needs: both sized at
      // INIT, so REQ allocates nothing.
      std::vector<uint8_t> mBytes;
      std::vector<float> mScratch;
      hw_program mProgram;
      bool mLoaded = false;
  };
} // namespace forte::handwired
