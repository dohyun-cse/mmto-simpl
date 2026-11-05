#pragma once
#include <iomanip>
#include <iostream>
#include <vector>
#include "mfem.hpp"

namespace mfem
{

/// Print a table of monitored variables, one row per call of Print().
class TableLogger
{
public:
   enum dtype { DOUBLE, INT };

protected:
   std::vector<double *> data_double;
   std::vector<int *> data_int;
   std::vector<dtype> data_order;
   std::vector<std::string> names;
   std::ostream &os;
   int w;
   bool var_name_printed;
   bool isRoot;
   std::unique_ptr<std::fstream> file;

public:
   /// Create a logger that prints to @a os.
   TableLogger(std::ostream &os = std::cout);
   void setw(const int column_width) { w = column_width; }
   /// Monitor the double variable @a val, referenced by address.
   void Append(const std::string name, double &val);
   /// Monitor the int variable @a val, referenced by address.
   void Append(const std::string name, int &val);
   /// Print a row with the current values of the monitored variables. The
   /// header row is printed on the first call, or if @a print_valname is true.
   void Print(bool print_valname=false);
   /// Also write each printed row to the file @a filename.
   void SaveWhenPrint(std::string filename,
                      std::ios::openmode mode = std::ios::out);
   void CloseFile() { if (file) { file.reset(nullptr); } }
};

/// Send grid functions to GLVis windows arranged in a grid.
class GLVis
{
   std::vector<std::unique_ptr<socketstream>> sockets;
   Array<mfem::GridFunction *> gfs;
   Array<mfem::QuadratureFunction *> qfs;
   Array<Mesh *> meshes;
   Array<bool> parallel;
   const char *hostname;
   const int port;
   int w, h, nrWinPerRow;
   bool secure;

public:
#ifdef MFEM_USE_GNUTLS
   static const bool secure_default = true;
#else
   static const bool secure_default = false;
#endif
   GLVis(const char hostname[], int port, int w=400, int h=350,
         int nrWinPerRow=1,
         bool secure = secure_default)
      : sockets(0), gfs(0), meshes(0), parallel(0), hostname(hostname),
        port(port), w(w), h(h), nrWinPerRow(nrWinPerRow),
        secure(secure_default) {}

   /// Open a window for @a gf with the given title and GLVis keys.
   void Append(GridFunction &gf, const char window_title[] = nullptr,
               const char keys[] = nullptr);
   void Append(QuadratureFunction &qf, const char window_title[] = nullptr,
               const char keys[] = nullptr);
   /// Send the current data to all windows.
   void Update();

   GridFunction& GetGridFunction(int i)
   {
      MFEM_VERIFY(i < gfs.Size(), "Index out of range");
      return *gfs[i];
   }

   socketstream &GetSocket(int i)
   {
      MFEM_VERIFY(i < sockets.size(), "Index out of range");
      return *sockets[i];
   }
};

} // namespace mfem
