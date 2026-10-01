/*
    Modified by AF 11/20/2026
    Post-hoc String Method (PHSM) (Mahmoud Moradi, 1/31/2014)
    Finds the transition tube between two conformations in a multidimensional free energy map.
sigma: for scaling the Euclidean distance
delta: is the max. radius
epsilon: is a margin for clossness to a center
bspline parameters: gravity, complexity
accuracy: for convergence

THICKNESS limits how far a point may stray along the path direction when being assigned, keeping clusters locally "orthogonal" to the string.
Smaller THICKNESS = stricter “thin tube” around each image; larger THICKNESS = accept elongated, along-path scatter.
delta: radial cutoff in the same metric as above. A point is only eligible for its nearest image if its distance D satisfy D < delta and is separated from the runner-up by epsilon. This keeps the string from “pulling” on very distant points; tune up if you’re rejecting too many reasonable points, down if distant noise is being included.

*/

#include <stdio.h>
#include <ctype.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <regex>

using namespace std;

#define boltzman 0.001987191683

namespace {

    int T = 10; // number of images
    int T1 = 100; // number of raw images

    int R = 1000; // Maximum number of iterations (to avoid freezing)
    double accuracy = 0.1; // iterates until the deviation converges up to this value

    double MAX = numeric_limits<double>::max();

    double temperature = 300.0;
    int input = 0; // weight(0) or free energy(1) as input
    
    //General variables
    int n = 1; // number of dimensions
    int N = 1; // number of data points
    
    double* w; // weights
    double* X; // collective variables (used for measuring the distance)
    int* c; // clusters
    double* C; // centerline

    // Distance Measure
    double* sigma; // the widths of coordinates (related to the metric)
    int* cvtype; // the widths of coordinates (related to the metric; 1,2,3,4 for quaternion elements; 0 for other things)
    int fix_endpoints = 0; // optional fixed reactant/product coordinates
    int warm_start = 0; // use an existing T-image path without initial smoothing
    bool q_euclidean = false; // use normalized quaternion components instead of angles

    // PHSM algorithm variables
    double epsilon = 0.01; // smallest recognizable difference between distances
    double delta = 1.0; // largest allowable distance from a center
    double thickness = 0.1; // thickness of the hyperplanes
    int orth = 1; // orthogonality method 0:projection 1:tangent
    
    // bspline
    double gravity = 1.0;
    int complexity = 1;

    // progress
    int monotonic = -1; // progress is (negative:not) monotonic (relative to product(0),reactant(1),both(2))
    double tolerance = 0.0; // the distance tolerated to deviate from monotonic move towards the product
        
    // Output filenames
    string clusterfile = "cluster.npy";
    string pathfile = "path.txt"; // filename to report the path and the free energies
    string smoothpathfile = "smoothpath.txt"; // filename to report the smooth path and the free energies

    static void unit_quat(const double* q, double* unit) {
        double norm = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            std::cerr << "Invalid zero or non-finite quaternion.\n";
            std::exit(1);
        }
        for (int k = 0; k < 4; k++) unit[k] = q[k] / norm;
    }

    static double quat_dot(const double* q, const double* p) {
        return q[0]*p[0] + q[1]*p[1] + q[2]*p[2] + q[3]*p[3];
    }

    // Align to a local reference before component averaging/interpolation.
    // A global q0 >= 0 convention introduces a discontinuity at q0 = 0.
    void align_quaternions(double* cvs, const double* reference) {
        for (int i = 0; i + 3 < n; i++) {
            if (cvtype[i] == 1) {
                if (quat_dot(cvs + i, reference + i) < 0.0)
                    for (int k = 0; k < 4; k++) cvs[i+k] = -cvs[i+k];
                i += 3;
            }
        }
    }

    double* normalize(double* cvs) {
    	for (int i = 0; i < n; i++) {
        	if (cvtype[i] == 1) {  // quaternion starts at i: (i,i+1,i+2,i+3)
            	if (i + 3 >= n) break;  // safety

                unit_quat(cvs + i, cvs + i);
            	i += 3; // skip the next 3 entries (part of this quaternion)
        	}
        	// else: non-quaternion CVs left unchanged
    	}
    	return cvs;
    }

    //double* normalize(double* cvs) {
    //    double norm;
    //    for (int i=0; i<n; i++) {
    //	    if (cvtype[i]==1) {
    //    	norm = 0;
    //    	for (int ii=i; ii<i+4; ii++)
    //    	    norm+=cvs[ii]*cvs[ii];
    //	    }
    //        if (cvtype[i]>0&&norm>0) cvs[i] /= sqrt(norm);
    //    }
    //    return cvs;
    //}
    //

    static inline double quat_angle(const double* q, const double* p) {
        double uq[4], up[4];
        unit_quat(q, uq);
        unit_quat(p, up);
        double dot = quat_dot(uq, up);
    	dot = std::abs(dot);
    	dot = std::min(1.0, std::max(0.0, dot));
    	return std::acos(dot); // radians
    }

    double quat_distance_squared(const double* q, const double* p, int offset) {
        if (q_euclidean) {
            double uq[4], up[4], total = 0.0, opposite = 0.0;
            unit_quat(q, uq);
            unit_quat(p, up);
            for (int k = 0; k < 4; k++) {
                double d = (uq[k]-up[k])/sigma[offset+k];
                total += d*d;
                d = (uq[k]+up[k])/sigma[offset+k];
                opposite += d*d;
            }
            return std::min(total, opposite);
        }
        double theta = quat_angle(q, p);
        return (theta*theta)/(sigma[offset]*sigma[offset]);
    }

    // distance between data and string image
    //double dist(int a, int b) {
    //    double tot = 0;
    //    for (int i=0; i<n; i++)
    //        tot+=((X[a*n+i]-C[b*n+i])*(X[a*n+i]-C[b*n+i]))/(sigma[i]*sigma[i]);
    //    return sqrt(tot);
    //}
    double dist(int a, int b) {
        double tot = 0.0;
        for (int i = 0; i < n; i++) {
                if (cvtype[i] == 1) {
                        if (i + 3 >= n) break;
                        const double* qa = &X[a*n + i];
                        const double* qb = &C[b*n + i];
                        tot += quat_distance_squared(qa, qb, i);
                        i += 3; // skip remaining quaternion components
                } else {
                        double d = (X[a*n + i] - C[b*n + i]) / sigma[i];
                        tot += d*d;
                }
        }
        return std::sqrt(tot);
    }
    // distance between two string image
    //double distC(int a, int b) {
    //    double tot = 0;
    //    for (int i=0; i<n; i++)
    //        tot+=((C[a*n+i]-C[b*n+i])*(C[a*n+i]-C[b*n+i]))/(sigma[i]*sigma[i]);
    //    return sqrt(tot);
    //}
    double distC(int a, int b) {
    	double tot = 0.0;
    	for (int i = 0; i < n; i++) {
        	if (cvtype[i] == 1) {
        	        if (i + 3 >= n) break;
        	        const double* qa = &C[a*n + i];
        	        const double* qb = &C[b*n + i];
	        tot += quat_distance_squared(qa, qb, i);
            		i += 3; // skip remaining quaternion components
        	} else {
            		double d = (C[a*n + i] - C[b*n + i]) / sigma[i];
            		tot += d*d;
        	}
    	}
    	return std::sqrt(tot);
    }
    double distAB(double* A, double* B, int a, int b);
    // Distance between path images in the same metric as clustering.
    double distA(double* A, int a, int b) {
        return distAB(A, A, a, b);
    }
    // distance from a point to the segment between two images (orthogonal distance)
    //double distAB(double* A, double* B, int a, int b) {
    //    double tot = 0;
    //    for (int i=0; i<n; i++)
    //        tot+=((A[a*n+i]-B[b*n+i])*(A[a*n+i]-B[b*n+i]))/(sigma[i]*sigma[i]);
    //    return sqrt(tot);
    //}
    double distAB(double* A, double* B, int a, int b) {
        double tot = 0.0;
        for (int i = 0; i < n; i++) {
                if (cvtype[i] == 1) {
                        if (i + 3 >= n) break;
                        const double* qa = &A[a*n + i];
                        const double* qb = &B[b*n + i];
                        tot += quat_distance_squared(qa, qb, i);
                        i += 3; // skip remaining quaternion components
                } else {
                        double d = (A[a*n + i] - B[b*n + i]) / sigma[i];
                        tot += d*d;
                }
        }
        return std::sqrt(tot);
    }
    // Shortest quaternion displacement in the tangent space at base.
    // Its length is acos(|base . point|), in radians (Colvars convention).
    void quat_log(const double* base, const double* point, double* v) {
        double q[4], p[4];
        unit_quat(base, q);
        unit_quat(point, p);
        double dot = quat_dot(q, p);
        if (dot < 0.0) {
            for (int k = 0; k < 4; k++) p[k] = -p[k];
            dot = -dot;
        }
        dot = std::min(1.0, dot);
        double sin2 = 0.0;
        for (int k = 0; k < 4; k++) {
            v[k] = p[k] - dot*q[k];
            sin2 += v[k]*v[k];
        }
        double sine = std::sqrt(sin2);
        double scale = sine > 1e-15 ? std::atan2(sine, dot)/sine : 0.0;
        for (int k = 0; k < 4; k++) v[k] *= scale;
    }

    double along_path(int a, int b, int left, int right) {
        double tot = 0.0, length2 = 0.0;
        for (int i = 0; i < n; i++) {
            if (cvtype[i] == 1) {
                if (q_euclidean) {
                    double sx = quat_dot(X+a*n+i, C+b*n+i) < 0 ? -1 : 1;
                    double sl = quat_dot(C+left*n+i, C+b*n+i) < 0 ? -1 : 1;
                    double sr = quat_dot(C+right*n+i, C+b*n+i) < 0 ? -1 : 1;
                    for (int k = 0; k < 4; k++) {
                        double dx = (sx*X[a*n+i+k]-C[b*n+i+k])/sigma[i+k];
                        double tangent = (sr*C[right*n+i+k]-sl*C[left*n+i+k])/sigma[i+k];
                        tot += dx*tangent;
                        length2 += tangent*tangent;
                    }
                } else {
                    double dx[4], dl[4], dr[4];
                    quat_log(C+b*n+i, X+a*n+i, dx);
                    quat_log(C+b*n+i, C+left*n+i, dl);
                    quat_log(C+b*n+i, C+right*n+i, dr);
                    for (int k = 0; k < 4; k++) {
                        double tangent = (dr[k]-dl[k])/sigma[i];
                        tot += (dx[k]/sigma[i])*tangent;
                        length2 += tangent*tangent;
                    }
                }
                i += 3;
            } else {
                double dx = (X[a*n+i]-C[b*n+i])/sigma[i];
                double tangent = (C[right*n+i]-C[left*n+i])/sigma[i];
                tot += dx*tangent;
                length2 += tangent*tangent;
            }
        }
        return length2 > 0.0 ? std::abs(tot)/std::sqrt(length2) : 0.0;
    }

    double proj(int a, int b) {
        return along_path(a, b, b > 0 ? b-1 : b, b < T-1 ? b+1 : b);
    }

    double tang(int a, int b) {
        return along_path(a, b, b > 0 ? b-1 : 0, b > 0 ? b : 1);
    }


    int assign(int s) {
    	    int index = -1;
    	    double D1 = MAX;
    	    double D2 = MAX;
    	    for (int t=0; t<T; t++) {
    		double D = dist(s,t);
    		if(D<D1) {
    		    index = t;
    		    D2=D1;
    		    D1=D;
    		} else if (D<D2) {
    		    D2=D;
    		}
    	    }
    	    int accept = 0;
    	    if (D2-D1>epsilon&&D1<delta) {
    		if (monotonic<0) {
    		    accept = 1;
    		} else if (monotonic==0) {
    		    if (index==T-1)
    			accept = 1;
    		    else if (dist(s,T-1)-distC(index+1,T-1)>-tolerance)
    			accept = 1;
    		} else if (monotonic==1) {
    		    if (index==0)
    			accept = 1;
    		    else if (dist(s,0)-distC(index-1,0)>-tolerance)
    			accept = 1;
    		} else if (monotonic==2) {
    		    if (index==T-1) {
    			if (dist(s,0)-distC(index-1,0)>-tolerance)
    			    accept = 1;
    		    } else if (index==0) {
    			if (dist(s,T-1)-distC(index+1,T-1)>-tolerance)
    			    accept = 1;
    		    } else 
    			if (dist(s,0)-distC(index-1,0)>-tolerance)
    			    if (dist(s,T-1)-distC(index+1,T-1)>-tolerance)
    				accept = 1;
    		}
            }
    	    if (accept) {
    		if (orth==0) {
    		    if (proj(s,index)>thickness)
    			accept=0;
    		} else if (orth==1) {
    		    if (tang(s,index)>thickness)
    			accept=0;
    		}
    	    }
        return accept ? index : -1;
    }

    // Rational Bezier Spline Curve algorithm to re-parametrize the path

    void bspline(double* Ci, int* poles, int Ni) {

        for (int t = 0; t < Ni; t++) {
            normalize(Ci + t*n);
            if (t > 0)
                align_quaternions(Ci + t*n, Ci + (t-1)*n);
        }

	double* xx = new double[n];
	double* xx_ = new double[n];
	double* Px = new double[Ni*n];
	double* W = new double[Ni];

	for (int i=0; i<n; i++) {
	    xx[i] = Ci[i];
	}

	for (int seg=0; seg<T; seg++) {

	    double a = seg*1.0/(T-1);

	    for (int i = 0; i < n; i++)
		xx_[i] = xx[i];

	    for (int t=0; t<Ni; t++) {
		if (poles[t])
		    W[t] = gravity;
		else 
		    W[t] = 1.0;
		for (int i = 0; i < n; i++)
		    Px[t*n+i] = W[t] * Ci[t*n+i];
	    }

	    for (int tt = Ni-1; tt > 0; tt--)
    		for (int t = 0; t < tt; t++) {
    		    for (int i = 0; i<n; i++)
        		Px[t*n+i] = (1-a)*Px[t*n+i] + a*Px[(t+1)*n+i];
        	    W[t] = (1-a)*W[t] + a * W[t+1];
        	}
    	    for(int i = 0; i<n; i++) {
		xx[i] = Px[i]/W[0];
		C[seg*n+i] = xx[i];
	    }
            normalize(C + seg*n);
	}

	delete[] xx;
	delete[] xx_;
	delete[] Px;
	delete[] W;
    }


    void npy_error(const string& message) {
        std::cerr << "NumPy: " << message << "\n";
        std::exit(1);
    }

    bool little_endian() {
        const uint16_t value = 1;
        return *reinterpret_cast<const unsigned char*>(&value) == 1;
    }

    int read_npy_header(std::istream& file, size_t rows, size_t columns, bool& swap_bytes) {
        unsigned char prefix[8];
        if (!file.read(reinterpret_cast<char*>(prefix), 8) ||
            std::memcmp(prefix, "\x93NUMPY", 6) != 0 ||
            prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0)
            npy_error("invalid or unsupported file header.");
        unsigned char length_bytes[4] = {};
        int length_size = prefix[6] == 1 ? 2 : 4;
        if (!file.read(reinterpret_cast<char*>(length_bytes), length_size))
            npy_error("truncated header.");
        uint32_t length = 0;
        for (int i = 0; i < length_size; i++)
            length |= uint32_t(length_bytes[i]) << (8*i);
        if (length == 0 || length > 1048576)
            npy_error("invalid header length.");
        string header(length, '\0');
        if (!file.read(&header[0], length)) npy_error("truncated header.");
        std::smatch match;
        if (!std::regex_search(header, match,
                std::regex("['\"]descr['\"]\\s*:\\s*['\"]([<>=|])f([48])['\"]")))
            npy_error("data must have float32 or float64 dtype.");
        char endian = match[1].str()[0];
        int width = match[2].str()[0] - '0';
        if (endian == '|') npy_error("floating-point dtype must specify byte order.");
        swap_bytes = endian != '=' && ((endian == '<') != little_endian());
        if (!std::regex_search(header, match,
                std::regex("['\"]fortran_order['\"]\\s*:\\s*(True|False)")) ||
            match[1] != "False")
            npy_error("data must be C-contiguous; use numpy.ascontiguousarray before saving.");
        if (!std::regex_search(header, match,
                std::regex("['\"]shape['\"]\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,?\\s*\\)")))
            npy_error("data must be a two-dimensional array.");
        try {
            if (std::stoull(match[1].str()) != rows ||
                std::stoull(match[2].str()) != columns)
                npy_error("shape must match (--points, --dimension + 1).");
        } catch (const std::exception&) {
            npy_error("invalid array dimensions.");
        }
        return width;
    }

    void read_npy_row(std::istream& file, double* values, size_t columns,
                      int width, bool swap_bytes) {
        for (size_t i = 0; i < columns; i++) {
            char bytes[8];
            if (!file.read(bytes, width)) npy_error("truncated data.");
            if (swap_bytes) std::reverse(bytes, bytes + width);
            if (width == 4) {
                float value;
                std::memcpy(&value, bytes, 4);
                values[i] = value;
            } else {
                std::memcpy(&values[i], bytes, 8);
            }
            if (!std::isfinite(values[i])) npy_error("data must contain finite values.");
        }
    }

    void write_npy_header(std::ostream& file, size_t rows) {
        string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
            std::to_string(rows) + ", 2), }";
        header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
        header += '\n';
        unsigned char prefix[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
            static_cast<unsigned char>(header.size() & 255),
            static_cast<unsigned char>((header.size() >> 8) & 255)};
        file.write(reinterpret_cast<const char*>(prefix), 10);
        file.write(header.data(), header.size());
    }

    void write_npy_value(std::ostream& file, double value) {
        char bytes[8];
        std::memcpy(bytes, &value, 8);
        if (!little_endian()) std::reverse(bytes, bytes + 8);
        file.write(bytes, 8);
    }

    bool read_row(std::istream& file, double* values, int columns, const char* filename) {
        string line;
        while (std::getline(file, line)) {
            line = line.substr(0, line.find('#'));
            if (line.find_first_not_of(" \t\r") == string::npos) continue;
            std::istringstream row(line);
            for (int i = 0; i < columns; i++) {
                if (!(row >> values[i]) || !std::isfinite(values[i])) {
                    std::cerr << "Invalid row in " << filename << "; expected " << columns << " finite values.\n";
                    std::exit(1);
                }
            }
            string extra;
            if (row >> extra) {
                std::cerr << "Too many columns in " << filename << ".\n";
                std::exit(1);
            }
            return true;
        }
        return false;
    }

    // reading parameters from standard input
    // '()' gives the default value
    // the parameters will be written in the standard error
    template <typename T>
    void read_parameter(T& value, const string& key) {
        if (!(std::cin >> value)) {
            std::cerr << "Missing or invalid value for " << key << ".\n";
            std::exit(1);
        }
    }

    void readparm() {

        string key;
	
        std::cerr << "Reading the parameters ...\n";
        while (std::cin >> key) {
            if (!key.empty() && key[0] == '#') {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }
            if (key == "END") break;
            // TEMPERATURE (300.0)
            if (key=="TEMPERATURE") {
                read_parameter(temperature, key);
                std::cerr << "TEMPERATURE = " << temperature << "\n";
            // PATHFILE (path.txt)
            } else if (key=="PATHFILE") {
                read_parameter(pathfile, key);
                std::cerr << "Path Filename = " << pathfile << "\n";
            // FREEENERGY
            } else if (key=="FREEENERGY") {
        	input = 1;
                std::cerr << "Weights will be converted from free energies.\n";
            // SMOOTHPATHFILE (smoothpath.txt)
            } else if (key=="SMOOTHPATHFILE") {
                read_parameter(smoothpathfile, key);
                std::cerr << "Smooth Path Filename = " << smoothpathfile << "\n";
            } else if (key=="CLUSTERFILE") {
                read_parameter(clusterfile, key);
                std::cerr << "Cluster Filename = " << clusterfile << "\n";
            // Post-hoc String Method Parameters
            // SIGMA sigma_0 sigma_1 ... (1 1 ...) 
            } else if (key=="SIGMA") {
                for (int i=0; i<n; i++) {
                    read_parameter(sigma[i], key);
                    std::cerr << "Sigma_" << i << " = " << sigma[i] << "\n";
                }
            } else if (key=="CVTYPE") {
                for (int i=0; i<n; i++) {
                    read_parameter(cvtype[i], key);
                    std::cerr << "cvtype_" << i << " = " << cvtype[i] << "\n";
                }
            } else if (key=="FIX_ENDPOINTS") {
                read_parameter(fix_endpoints, key);
            } else if (key=="WARM_START") {
                read_parameter(warm_start, key);
            } else if (key=="Q_Euclidean") {
                q_euclidean = true;
                std::cerr << "Quaternion distance = normalized Euclidean components.\n";
            } else if (key=="MONOTONIC") {
                read_parameter(monotonic, key);
                if (monotonic==0)
            	    std::cerr << "product";
            	else if (monotonic==1)
            	    std::cerr << "reactant";
            	else if (monotonic==2)
            	    std::cerr << "product/reactant";
            	else std::cerr << "nothing";    
            	std::cerr << " will be the reference to determine the monotonic progress.\n";
            } else if (key=="TOLERANCE") {
                read_parameter(tolerance, key);
                std::cerr << tolerance << " deviation from monotonic move is tolerated.\n";
            } else if (key=="GRAVITY") {
                read_parameter(gravity, key);
                std::cerr << gravity << " will be used for poles.\n";
            } else if (key=="COMPLEXITY") {
                read_parameter(complexity, key);
                std::cerr << complexity << " will be used to scale the dimensionality of the spline.\n";
            } else if (key=="ITERATIONS") {
                read_parameter(R, key);
                std::cerr << R << " is the maximum number of iterations.\n";
            } else if (key=="IMAGES") {
                read_parameter(T, key);
                std::cerr << T << " is the number of points on the path.\n";
            } else if (key=="EPSILON") {
                read_parameter(epsilon, key);
                std::cerr << "epsilon = " << epsilon << "\n";
            } else if (key=="ACCURACY") {
                read_parameter(accuracy, key);
                std::cerr << "accuracy = " << accuracy << "\n";
            } else if (key=="ORTHOGONALITY") {
                read_parameter(orth, key);
                std::cerr << "orthogonality method = " << orth << "\n";
            } else if (key=="THICKNESS") {
                read_parameter(thickness, key);
                std::cerr << "thickness = " << thickness << "\n";
            } else if (key=="DELTA") {
                read_parameter(delta, key);
                std::cerr << "delta = " << delta << "\n";
            } else {
                std::cerr << "Unknown parameter: " << key << "\n";
                std::exit(1);
            }
        }
    }
}

// Main

int main(int argc,char *argv[]) {

    // Reading the arguments (file names)
    const char* datafile = NULL; // data file (data points)
    const char* guessfile = NULL; // path file (guesses)
    int argi = 0;
    
    std::cerr << "Parsing the arguments ...\n";
    while (++argi < argc) {
        std::string argstr(argv[argi]);
        if (argi+1 >= argc) {
            std::cerr << "Missing argument for " << argstr << "\n";
            return 1;
        }
        if ( argstr == "--Data" || argstr == "-D" ) {
            datafile = argv[++argi];
        } else if ( argstr == "--Guess" || argstr == "-G" ) {
            guessfile = argv[++argi];
        } else if ( argstr == "--dimension" || argstr == "-d" ) {
            n = atoi(argv[++argi]);
        } else if ( argstr == "--images" || argstr == "-i" ) {
            T1 = atoi(argv[++argi]);
        } else if ( argstr == "--points" || argstr == "-p" ) {
            N = atoi(argv[++argi]);
        }
    }
    
    std::cerr << "A "<< n <<"-dimensional weight map will be imported from " << datafile << " ( " << N << " data points ).\n";
    std::cerr << "A "<< n <<"-dimensional initial pathway will be imported from " << guessfile << " ( " << T1 << " data points ).\n";
    
    if (n < 1 || N < 1 || T1 < 2 || !datafile || !guessfile) {
        std::cerr << "Require data, guess, positive dimensions/points and at least two images.\n";
        return 1;
    }
    sigma = new double[n];
    cvtype = new int[n];
    for (int i=0;i<n;i++) {
	sigma[i]=1.0;
	cvtype[i]=0.0;
    }
    // Reading the parameters
    readparm();

    if (T < 2 || R < 0 || !std::isfinite(accuracy) || accuracy < 0 ||
        !std::isfinite(delta) || delta <= 0 || !std::isfinite(thickness) || thickness < 0 ||
        !std::isfinite(epsilon) || epsilon < 0 || !std::isfinite(temperature) || temperature <= 0 ||
        !std::isfinite(gravity) || gravity <= 0 || complexity < 0 || complexity > 2 ||
        (orth != 0 && orth != 1) || (fix_endpoints != 0 && fix_endpoints != 1) ||
        (warm_start != 0 && warm_start != 1)) {
        std::cerr << "Invalid PHSM parameters.\n";
        return 1;
    }

    // Reject malformed quaternion blocks and unusable metric scales.
    for (int i = 0; i < n; i++) {
        if (!(sigma[i] > 0.0) || !std::isfinite(sigma[i])) {
            std::cerr << "SIGMA must contain positive finite values.\n";
            return 1;
        }
        if (cvtype[i] != 0) {
            int start = i - (cvtype[i]-1);
            if (cvtype[i] < 1 || cvtype[i] > 4 || start < 0 || start+3 >= n ||
                cvtype[start] != 1 || cvtype[start+1] != 2 ||
                cvtype[start+2] != 3 || cvtype[start+3] != 4) {
                std::cerr << "CVTYPE quaternions must be complete 1 2 3 4 blocks.\n";
                return 1;
            }
        }
    }
    
    temperature *= boltzman;

    //
    // Importing the data
    //

    w = new double[N];
    X = new double[N*n];
    c = new int[N];
    C = new double[T*n];

    string data_name(datafile);
    bool npy_input = data_name.size() >= 4 &&
        data_name.substr(data_name.size()-4) == ".npy";
    ifstream wfile(datafile, std::ios::binary);
    if (!wfile) {
        std::cerr << "Cannot open data input file.\n";
        return 1;
    }
    bool swap_bytes = false;
    int npy_width = npy_input ? read_npy_header(wfile, N, size_t(n)+1, swap_bytes) : 0;
    vector<double> row(n+1);
    double total_weight = 0.0;
    for (int s = 0; s < N; s++) {
        if (npy_input) {
            read_npy_row(wfile, row.data(), size_t(n)+1, npy_width, swap_bytes);
        } else if (!read_row(wfile, row.data(), n+1, datafile)) {
            std::cerr << "Too few data rows in " << datafile << ".\n";
            return 1;
        }
        for (int i = 0; i < n; i++) X[s*n+i] = row[i];
        normalize(X+s*n);
        w[s] = input ? exp(-row[n]/temperature) : row[n];
        if (!std::isfinite(w[s]) || w[s] < 0.0) {
            std::cerr << "Weights must be finite and nonnegative.\n";
            return 1;
        }
        total_weight += w[s];
    }
    if ((npy_input ? wfile.peek() != std::char_traits<char>::eof() :
         read_row(wfile, row.data(), n+1, datafile)) ||
        !(total_weight > 0.0) || !std::isfinite(total_weight)) {
        std::cerr << "Check the data row count and total weight.\n";
        return 1;
    }
    std::cerr << N << " data points read.\n";

    double* Cii = new double[T1*n];
    int* Pii = new int[T1]();
    ifstream gfile(guessfile);
    for (int t = 0; t < T1; t++) {
        if (!read_row(gfile, Cii+t*n, n, guessfile)) {
            std::cerr << "Too few path rows in " << guessfile << ".\n";
            return 1;
        }
        normalize(Cii+t*n);
        if (t > 0) align_quaternions(Cii+t*n, Cii+(t-1)*n);
    }
    if (read_row(gfile, row.data(), n, guessfile)) {
        std::cerr << "Too many path rows in " << guessfile << ".\n";
        return 1;
    }
    std::cerr << T1 << " points on the path read.\n";

    // Reparametrize
    const char* PATHFILE = pathfile.c_str();
    ofstream path (PATHFILE);
    const char* SMOOTHPATHFILE = smoothpathfile.c_str();
    ofstream smoothpath (SMOOTHPATHFILE);

    if (!path || !smoothpath) {
        std::cerr << "Cannot open path output files.\n";
        return 1;
    }
    path << std::setprecision(17);
    smoothpath << std::setprecision(17);
    if (warm_start) {
        if (T1 != T) {
            std::cerr << "WARM_START requires exactly IMAGES input rows.\n";
            return 1;
        }
        std::copy(Cii, Cii+T*n, C);
    } else bspline(Cii,Pii,T1);
    
    delete[] Cii;
    delete[] Pii;
    
    int r = 0;
    double* m = new double[T]();
    int* M = new int[T];
    for (int t=0; t<T; t++) {
    	smoothpath << r << " " << t << " " << m[t] << " ";
        for (int i=0; i<n; i++)
            smoothpath << C[t*n+i] << " ";
        smoothpath << "\n";
    }

    // Iterative clustering

    std::cerr << "Clustering begins...\n";
    
    double* C_ = new double[T*n]; // future centerline
    double* C__ = new double[T*n]; // the old centerline

    double SD = MAX; // Mean image displacement from the old centerline
    while (r<R && SD>accuracy) {

	for (int t=0; t<T; t++) {
    	    m[t] = 0.0;
    	    M[t] = 0;
    	    for (int i=0; i<n; i++) C_[t*n+i] = 0.0;
    	}

    	for (int s=0; s<N; s++) {
            int index = assign(s);
            c[s] = index;
            if (index >= 0 && w[s] > 0.0) {
    	    	    m[index] += w[s];
    		    M[index] ++;
                    align_quaternions(X + s*n, C + index*n);
    		    for (int i=0; i<n; i++) {
    			C_[index*n+i] += X[s*n+i] * w[s];
    		    }	
            }
	}

        T1 = T;
        double accepted_weight = 0.0;
        for (int t = 0; t < T; t++) accepted_weight += m[t];
        if (!(accepted_weight > 0.0)) {
            std::cerr << "No positive weight inside the PHSM tube.\n";
            return 1;
        }
        for (int t = 0; t < T; t++) {
            bool fixed = fix_endpoints && (t == 0 || t == T-1);
            for (int i = 0; i < n; i++) {
                C__[t*n+i] = C[t*n+i];
                C_[t*n+i] = (m[t] > 0.0 && !fixed) ? C_[t*n+i]/m[t] : C[t*n+i];
            }
            if (!(m[t] > 0.0))
                std::cerr << "Step " << r << ": retaining zero-mass image " << t << ".\n";
            normalize(C_+t*n);
            if (t > 0) align_quaternions(C_+t*n, C_+(t-1)*n);
            path << r << " " << t << " " << (m[t] > 0.0 ? -temperature*log(m[t]) :
                std::numeric_limits<double>::infinity()) << " ";
            for (int i = 0; i < n; i++) path << C_[t*n+i] << " ";
            path << "\n";
        }
		
	double ds0=MAX;
	double tlen=0;
	for (int tt=0; tt<T1-1; tt++) {
	    double ds = distA(C_,tt,tt+1);
	    tlen+=ds;
	    if (ds<ds0) ds0=ds;
	}
        if (!(ds0 > 0.0) || !std::isfinite(tlen)) {
            std::cerr << "Coincident or invalid path images; cannot reparametrize.\n";
            return 1;
        }
		
	int Tmax = (complexity+1)*int(1+tlen/ds0);
	Cii = new double[Tmax*n];
	Pii = new int[Tmax];

	int dt = 0;
	for (int tt=0; tt<T1-1; tt++) {
	    double ds = distA(C_,tt,tt+1);
	    for (int i=0; i<n; i++) {
		Cii[dt*n+i] = C_[tt*n+i];
	    }
	    Pii[dt] = 1;
	    int dT=int(complexity*ds/ds0);
	    if (dT>1&&dt<Tmax-dT) {
		for (int dtt=dt+1; dtt<dt+dT; dtt++) {
		    for (int i=0; i<n; i++)
			Cii[dtt*n+i] = C_[tt*n+i] - (double(dtt-dt)/dT) * (C_[tt*n+i] - C_[(tt+1)*n+i]);
		    Pii[dtt]=0;
		}
		dt+=dT-1;
	    }
	    dt++;
	}
	
	if (dt>=Tmax) {
	    std::cerr << "Too many points! " << dt << " > " << Tmax << "\n"; 
	    exit(1);
	}
	
	for (int i=0; i<n; i++)
	    Cii[dt*n+i] = C_[(T1-1)*n+i];
        Pii[dt] = 1;
			
	bspline(Cii,Pii,dt+1);
	
	delete[] Cii;
	delete[] Pii;
	
        SD = 0.0;
        r++;
        for (int t = 0; t < T; t++) SD += distAB(C,C__,t,t);
        SD /= T;
        // Intermediate third columns describe the centroid-update masses.
        // The final block below is reassigned to the final smoothed path.
        if (r < R && SD > accuracy) {
            for (int t = 0; t < T; t++) {
                smoothpath << r << " " << t << " " << (m[t] > 0.0 ? -temperature*log(m[t]) :
                    std::numeric_limits<double>::infinity()) << " ";
                for (int i = 0; i < n; i++) smoothpath << C[t*n+i] << " ";
                smoothpath << "\n";
            }
        }
        std::cerr << "Iteration " << r << ": changed " << SD << " from last iteration.\n";
    }

    std::cerr << (SD <= accuracy ? "PHSM converged.\n" : "PHSM iteration limit reached.\n");
    std::cerr << "Final clusters being reported...\n";
    ofstream cluster(clusterfile.c_str(), std::ios::binary);
    if (!cluster) {
        std::cerr << "Cannot open cluster output file.\n";
        return 1;
    }
    write_npy_header(cluster, N);
    std::fill(m, m+T, 0.0);
    for (int s = 0; s < N; s++) {
        c[s] = assign(s);
        if (c[s] >= 0) m[c[s]] += w[s];
        write_npy_value(cluster, static_cast<double>(c[s]));
        write_npy_value(cluster, c[s] >= 0 ? dist(s,c[s]) :
            std::numeric_limits<double>::infinity());
    }
    // R=0 is useful for assigning a fixed path; keep a distinct final block.
    for (int t = 0; t < T; t++) {
        smoothpath << (r ? r : 1) << " " << t << " " << (m[t] > 0.0 ? -temperature*log(m[t]) :
            std::numeric_limits<double>::infinity()) << " ";
        for (int i = 0; i < n; i++) smoothpath << C[t*n+i] << " ";
        smoothpath << "\n";
    }
    ofstream status((smoothpathfile+".status").c_str());
    status << (SD <= accuracy ? "converged" : "iteration_limit") << " " << r << " "
           << std::setprecision(17) << SD << "\n";
    path.close();
    smoothpath.close();

    cluster.close();
    if (!cluster) {
        std::cerr << "Failed to write cluster output file.\n";
        return 1;
    }
        
    delete[] w;
    delete[] X;
    delete[] c;
    delete[] C;
    delete[] C_;
    delete[] C__;
    delete[] m;
    delete[] M;
    delete[] sigma;
    delete[] cvtype;

    std::cerr << "Done!\n";

    return 0;
}
