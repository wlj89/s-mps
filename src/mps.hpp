#ifndef MPS_HPP
#define MPS_HPP

#define EIGEN_DONT_PARALLELIZE
//could use MKL for matrix operations? 
#include <Eigen/Eigen>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <complex>
#include <iostream>
#include <assert.h>
#include <random>
#include <cmath>
#include <fstream>
//#include "mpo.hpp"

using namespace std;
using namespace Eigen;

template <typename T> class MPS
{
    /*
        Matrix product state class
        !!! Eigen is column-major by default !!!

        !!! order of pauli matrix !!!
        
        0 1 
        0 0 ~ this is a creation operator

        0    
        1   ~ spindown / |0> 
        
        1
        0  ~ spinup / |1>
    */
    
    typedef Matrix<T,Dynamic,Dynamic> Mat;
    typedef Matrix<T,Dynamic,1> ColVec;
    typedef Matrix<T,1,Dynamic> RowVec; 
    typedef Matrix<int,2,2> PauliMat;       // pauli matrices
    typedef vector<PauliMat> PauliStr; 

public:

    unsigned bond_dim = 0; // max bond dimensions
    unsigned site_num = 0; // number of spin orbitals
    unsigned d;            // local physical degree of freedom. For spin d=2 

    double norm;
    
    vector<vector<Mat>> mat_vec; // the vector container of all matrices
    /*
        Note: 
        1. might be more advantageous to use a single block of memeory
            (but how)
        
        2. Watch out the order 
            mat_vec[0] ~ spin-up matrices aka |1>
            mat_vec[1] ~ spin-down matrices aka |0>
    */

    // the defualt constructor 
    MPS(){}
    MPS(unsigned bd_val, 
        unsigned site_num_val,
        unsigned d_val)
    {
        
        initialize(bd_val,site_num_val,d_val);    
    }
    
    void from_direct_sum(const MPS& phi_1, const MPS& phi_2)
    {
        /*
            forming direct sum of two MPS
        */
        
        assert(phi_1.site_num == phi_2.site_num);
        assert(phi_1.d == phi_2.d);

        bond_dim = phi_1.bond_dim+phi_2.bond_dim; 
        site_num = phi_1.site_num;
        d = phi_1.d;

        //!!!!
        /*
        mat_vec.resize(d); 
        
        for(int i=0; i<d; i++)
            mat_vec[i].resize(site_num);
        */
        initialize(phi_1.bond_dim+phi_2.bond_dim, phi_1.site_num, phi_1.d); 

        for (int i=0; i< site_num; i++)
            for (int j=0; j<d; j++)
            {
                int new_rows = phi_1.mat_vec[j][i].rows() + phi_2.mat_vec[j][i].rows(); 
                int new_cols = phi_1.mat_vec[j][i].cols() + phi_2.mat_vec[j][i].cols(); 
                
                //cout << i <<endl;
                //cout << "new columns:"  << new_cols <<endl; 
                
                if (i == 0 )
                {
                    // longer row vector 
                    mat_vec[j][i].resize(1,new_cols);
                    mat_vec[j][i].setZero(); 
                    //cout << "beep\n"; 
                    mat_vec[j][i].topLeftCorner(1, phi_1.mat_vec[j][i].cols()) = phi_1.mat_vec[j][i];

                    mat_vec[j][i].bottomRightCorner(1, phi_2.mat_vec[j][i].cols()) = phi_2.mat_vec[j][i]; 
            
                }
                else if (i == site_num-1)
                {
                    // longer col vec
                    mat_vec[j][i].resize(new_rows,1);
                    mat_vec[j][i].setZero(); 

                    mat_vec[j][i].topLeftCorner(phi_1.mat_vec[j][i].rows(), 1) = phi_1.mat_vec[j][i];

                    mat_vec[j][i].bottomRightCorner(phi_2.mat_vec[j][i].rows(), 1) = phi_2.mat_vec[j][i]; 
            
                }
                else 
                {
                    // block diagonal matrix
                    mat_vec[j][i].resize(new_rows,new_cols);
                    mat_vec[j][i].setZero(); 
                    
                    mat_vec[j][i].topLeftCorner(phi_1.mat_vec[j][i].rows(), phi_1.mat_vec[j][i].cols()) = phi_1.mat_vec[j][i];

                    mat_vec[j][i].bottomRightCorner(phi_2.mat_vec[j][i].rows(), phi_2.mat_vec[j][i].cols()) = phi_2.mat_vec[j][i]; 
            
                }

                         
            }
    }

    // copy constructor 
    void initialize(unsigned bd_val, 
                    unsigned site_num_val,
                    unsigned d_val) 
    {
        bond_dim = bd_val;
        site_num = site_num_val;
        d = d_val; 
        
        unsigned num_row = 1; 
        unsigned num_col = d;
        
        mat_vec.resize(d); 
        
        for(int i=0; i<d; i++)
            mat_vec[i].resize(site_num);
        
        // set bond dims 
        for(int i=0; i<site_num/2; i++)
        {
            /*
                site_num must be even;
                Notice that Schmidt num between the first site and the rest cannot be 
                greater than d 
                thus : (1,d) (d,d^2) .. (d^?, D) (D,D) .. ... (d,1)
            */  
            
            for(int j=0; j<d; j++)
            {   
                mat_vec[j][i].resize(num_row,num_col);
                mat_vec[j][site_num-1-i].resize(num_col,num_row);
            }
            if (num_row < bond_dim)
                num_row = num_row * d < bond_dim ? num_row *d : bond_dim; 

            if (num_col < bond_dim)
                num_col = num_col * d < bond_dim ? num_col *d : bond_dim; 
        }
    }

    void read(string wf_name)
    {
        
        // some safegaurd mechanism?

        const char* u = &wf_name[0];
        FILE *fp =  fopen(u,"r");
        
        fscanf(fp, "%d", &site_num);
        fscanf(fp, "%d", &d);

        for (int i=0; i<site_num; i++)
        {
            int num_row, num_col;
            fscanf(fp, "%d", &num_col);
            fscanf(fp, "%d", &num_row);

            for (int j=0; j < d; j++)
            {
                for (int row_idx = 0; row_idx < num_row; row_idx++)
                {
                    for (int col_idx=0; col_idx < num_col; col_idx++)
                        fscanf(fp, "%lf", &(mat_vec[j][i](row_idx,col_idx)));
                }
            }
        }
        
        fclose(fp);
        /*
        for(int i=0; i< site_num; i++)
        {
            cout << mat_vec[0][i] <<endl;
            cout << mat_vec[1][i] <<endl;
            cout << endl; 
        }*/
    }
    void write(string wf_name)
    {
        ofstream output(wf_name);
        output.precision(12);
        // # of site 
        output<< site_num <<endl; 
        
        // d 
        output << d <<endl; 

        for (int i = 0; i<site_num; i++)
        {
            // columns
            int num_cols = mat_vec[0][i].cols(); 
            int num_rows = mat_vec[0][i].rows(); 

            output << mat_vec[0][i].cols()<<endl; 
            output << mat_vec[0][i].rows()<<endl;

            for (int j = 0; j < d; j++)
            {
                for (int row_idx = 0; row_idx < num_rows; row_idx++)
                {
                    for (int col_idx=0; col_idx < num_cols; col_idx++)
                    {
                        output << mat_vec[j][i](row_idx, col_idx) <<" ";
                    }
                    output << endl;     
                }
            } 
        }

        output.close(); 
    }
    
    void blank_copy(const MPS& rhs)
    {
        bond_dim = rhs.bond_dim;
        site_num = rhs.site_num;
        d = rhs.d;

        mat_vec.resize(d);

        for(int i=0; i< d; i++)
        {
            mat_vec[i].resize(site_num);

            for (int j= 0; j< site_num; j++)
            {
                mat_vec[i][j].resize(rhs.mat_vec[i][j].rows(), rhs.mat_vec[i][j].cols());
                mat_vec[i][j] = rhs.mat_vec[i][j];
            
            }
        }
    }
    
    void careless_copy(const MPS& rhs)
    {
        /*
            no check on bd, site num, and d
            hence the name 
        */
        for(int i=0; i< d; i++)
        {
            mat_vec[i].resize(site_num);

            for (int j= 0; j< site_num; j++)
            {
                //mat_vec[i][j].resize(rhs.mat_vec[i][j].rows(), rhs.mat_vec[i][j].cols());
                mat_vec[i][j] = rhs.mat_vec[i][j];
                
            }
        }
    }

    T inner_product(const MPS& rhs)
    {
        /*
            calc <phi(self) | psi(rhs)>
            !!! in principle phi and psi can have different bd 
        */
        T val = 0;

        assert(this->site_num == rhs.site_num);
        assert(this->d == rhs.d);
        // only allocate buffer space once
        unsigned max_bd = max(this->bond_dim,rhs.bond_dim);

        Mat buff = Mat::Zero(max_bd,max_bd);
        Mat another_buff = Mat::Zero(max_bd,max_bd);
        
        unsigned new_cols, new_rows, old_cols, old_rows;

        old_rows = this->mat_vec[0][0].cols();
        old_cols = rhs.mat_vec[0][0].cols();

        for (int j=0; j < d; j++)
        {
            buff.topLeftCorner(old_rows,old_cols) += this->mat_vec[j][0].adjoint() * rhs.mat_vec[j][0]; 
        }   

        for (int i=1; i< this->site_num; i++)
        {
            new_rows = this->mat_vec[0][i].cols(); 
            new_cols = rhs.mat_vec[0][i].cols();

            another_buff.topLeftCorner(new_rows,new_cols).setZero();
            
            for (int j = 0; j < d; j++)
            {
                another_buff.topLeftCorner(new_rows,new_cols) 
                        += this->mat_vec[j][i].adjoint() * buff.topLeftCorner(old_rows,old_cols) * rhs.mat_vec[j][i];
            }

            buff.topLeftCorner(new_rows,new_cols) = another_buff.topLeftCorner(new_rows,new_cols);
            
            /*
            if (i==2)
            {
                cout << "L from inner product:\n";
                cout <<  buff.topLeftCorner(new_rows,new_cols)  <<endl;
            }
            */
            //cout << i <<endl;
            //cout << buff.topLeftCorner(new_rows,new_cols) <<endl<<endl; 
            
            old_rows = new_rows;
            old_cols = new_cols;
        }
        
        //cout << buff(0,0);
        return buff(0,0); 
    }
    
    T get_norm()
    {
        /*
            compute norm of a MPS
            Use inner_product(rhs) 
        */  
        return this->inner_product(*this);

    }  

    void normalize()
    {
        /*
            Not to be confused with on-site normalization 
            
        */
        double norm_val = this->inner_product(*this);
        norm_val = sqrt(norm_val);

        // the choice of matrix to be scaled down is pretty arbitrary
        this->mat_vec[0][0] /= norm_val;
        this->mat_vec[1][0] /= norm_val;
    }
    
    void right_canonicalize(unsigned left_pos, unsigned right_pos)
    {   
        /*
            transform a MPS to left canonical form
            for now only use Householder QR 

            NOTE: start from right_pos and end at left_pos+1 
        */
        HouseholderQR<Mat> q;

        for (int i=right_pos; i>left_pos; i--) 
        {
            unsigned site_cols = mat_vec[0][i].cols();
            unsigned site_rows = mat_vec[0][i].rows();
            
            // buffer for QR
            Mat buff(site_rows, d * site_cols);
            buff.setZero();
            
            for(int j = 0; j < d; j++)
            {
                buff.middleCols(j*site_cols, site_cols) = mat_vec[j][i];
            } 
                        
            q.compute(buff.adjoint());

            Mat Q = q.householderQ();

            if (buff.cols() >= buff.rows()) 
                Q = Q.leftCols(site_rows).eval();

            Mat R; 

            if (buff.cols() >= buff.rows()) 
                R = Mat(q.matrixQR().template triangularView<Upper>()).topLeftCorner(site_rows,site_rows); 
            else 
                R = Mat(q.matrixQR().template triangularView<Upper>()); 

            for(int j=0; j< d; j++)
            {
                mat_vec[j][i] = (Q.adjoint()).middleCols(j*site_cols,site_cols); 
                mat_vec[j][i-1] = mat_vec[j][i-1] * (R.adjoint()); 
            }
        }
    }
    
    void left_canonicalize(unsigned left_pos, unsigned right_pos)
    {
        /*
            transform a MPS to left canonical form
            for now only use Householder QR 
        */
        
        HouseholderQR<Mat> q;
        
        for(int i=left_pos; i< right_pos; i++)
        {
            unsigned site_cols = mat_vec[0][i].cols();
            unsigned site_rows = mat_vec[0][i].rows();

            Mat buff(d*mat_vec[0][i].rows(), mat_vec[0][i].cols());
            buff.setZero();
            
            //cout << "in left canonical:" << i <<endl;
            // collect M on site 
            for (int j =0; j < d ; j++)
            {
                buff.middleRows(j * site_rows, site_rows) = mat_vec[j][i];
            }
            
            //cout << "beep1" <<endl;
            q.compute(buff);
            
            /*
                Let A by a m*n matrix 
                when  m >= n, the following routine works 
                But in compression we encounter m < n 

                !!! when compressing mps  this might not hold !!! 
            */
            // get Q 
            Mat Q = q.householderQ();
            // thin Q 
            if (buff.cols() <= buff.rows() )
                Q = Q.leftCols(site_cols).eval();
            
            // get R
            // bizarre syntax is required here to enable templating
            Mat R; 
            if (buff.cols() <= buff.rows() )
                R = Mat(q.matrixQR().template triangularView<Upper>()).topLeftCorner(site_cols,site_cols); 
            else 
                R = Mat(q.matrixQR().template triangularView<Upper>()); 
            //cout << "beep2" <<endl;
            
            // put things back and apply R on the next site
            // automatically take care of the changed col number 
            for(int j=0; j < d; j++)
            {
                mat_vec[j][i] =  Q.middleRows(j * site_rows, site_rows);
                mat_vec[j][i+1] = R * mat_vec[j][i+1];
            }       
        }
    }

    void check_right_identity()
    {
        for(int i=0; i< site_num; i++)
        {
            cout << i <<endl;
            auto M =  mat_vec[0][i] * mat_vec[0][i].adjoint() + mat_vec[1][i] * mat_vec[1][i].adjoint();
            auto q = M.isIdentity(1E-12);
            cout <<q << endl<< endl; 
        }

    }

    bool check_mixed_canonical(int site_l)
    {
        //cout << "Left:\n";
        for (int i=0; i< site_l; i++)
        {
            //cout << i <<":";
            auto M =  mat_vec[0][i].adjoint() * mat_vec[0][i] + mat_vec[1][i].adjoint() * mat_vec[1][i];
            auto q = M.isIdentity(1E-12);
            if (q!=1) return false;
            
        }
        //cout << "Right:\n";
        for (int i=site_l+1; i < mat_vec[0].size(); i++)
        {
            //cout << i <<":";
            auto M =  mat_vec[0][i] * mat_vec[0][i].adjoint() + mat_vec[1][i] * mat_vec[1][i].adjoint();
            auto q = M.isIdentity(1E-12);
            if (q!=1) return false;
        }
        return true;
    }
    
    void compress_svd(int target_bd)
    {   
        /*
            target bond dimension
            
            no need to worry about the normalization when used for walkers         
        */
        JacobiSVD<MatrixXd> svd;
        
        right_canonicalize(0,site_num-1);

        for (int i=0; i< site_num-1; i++)
        {
            if  ( ((2 * mat_vec[0][i].rows() >= mat_vec[0][i].cols() ) && (mat_vec[0][i].cols() > target_bd)) 
                    ||
                  ((2 * mat_vec[0][i].rows() <  mat_vec[0][i].cols() ) && (2 * mat_vec[0][i].rows() > target_bd))
                )
            {
                // SVD
                //cout << "decomposing site " << i <<endl;
                //cout << "num of rows:" << mat_vec[0][i].rows() <<endl; 
                //cout << "num of cols:" << mat_vec[0][i].cols() <<endl; 

                int num_rows = mat_vec[0][i].rows(); 
                int num_cols = mat_vec[0][i].cols(); 
                
                Mat M(2*num_rows, num_cols);
                M.topLeftCorner(num_rows,num_cols) = mat_vec[0][i];
                M.bottomLeftCorner(num_rows,num_cols) = mat_vec[1][i];
                
                svd.compute(M, ComputeThinU | ComputeThinV);

                //cout << "Schmidt coeffs:\n";
                //cout << svd.singularValues() <<endl; 

                // throwing things away 
                // any dimensionality issue? 
                Mat U_trunc = svd.matrixU().topLeftCorner(2*num_rows,target_bd);
                Mat Sigma_trunc = Mat(svd.singularValues().head(target_bd).asDiagonal());
                Mat V_dagger_trunc = svd.matrixV().topLeftCorner(num_cols,target_bd).adjoint();

                mat_vec[0][i] = U_trunc.topRows(num_rows);
                mat_vec[1][i] = U_trunc.bottomRows(num_rows); 

                Mat prod = Sigma_trunc * V_dagger_trunc; 
                
                /*
                cout << "shape of sigma * V_dagger:\n";
                cout << prod.rows() << "," << prod.cols() <<endl; 
                cout << "shape of next:\n";
                cout << mat_vec[0][i+1].rows() << "," << mat_vec[0][i+1].cols() <<endl; 
                cout << mat_vec[1][i+1].rows() << "," << mat_vec[1][i+1].cols() <<endl; 
                */
                auto tmp_0 = prod * mat_vec[0][i+1];
                auto tmp_1 = prod * mat_vec[1][i+1];
                //cout << tmp <<endl; 

                mat_vec[0][i+1] = tmp_0;
                mat_vec[1][i+1] = tmp_1;
                
                //break; 
                //cout << endl; 
            }
            else
            {
                left_canonicalize(i,i+1);
            }

        }   
        
        bond_dim = target_bd; 
        //for (auto mat: mat_vec[0])
        //    cout << mat.rows() << "," << mat.cols() <<endl;
    }
    
    void rand_init()    
    {
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j< site_num; j++)
            {
                mat_vec[i][j] = Mat::Random(mat_vec[i][j].rows(),mat_vec[i][j].cols()); 
            }

        }   
    }
    
    void set_random()
    {
        // use c++11 engine rather than srand 
        std::random_device rd;
        std::mt19937_64 gen(123123);
        std::uniform_real_distribution<double> gen_dist(-1,1);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j< site_num; j++)
            {
                mat_vec[i][j] = Mat::NullaryExpr(mat_vec[i][j].rows(), mat_vec[i][j].cols(),
                                                    [&](){return gen_dist(gen);});
            }
        }   

        this->normalize();

    }

    void set_zero()
    {
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j< site_num; j++)
            {
                mat_vec[i][j].setZero(); 
            }
        }   

    }

    void print()
    {
        for (int i=0; i< site_num; i++)
        {
            cout << "site:" << i <<endl; 
            for (int j=0; j< d; j++)
            {
                cout << mat_vec[j][i] <<endl<<endl; 
            }
            //cout << 
        }
    }
    void set_uniform()
    {
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j< site_num; j++)
            {
                mat_vec[i][j] = Mat::Constant(mat_vec[i][j].rows(),mat_vec[i][j].cols(),1);
            }

        } 
    }
    void get_R_mat( unsigned l,
                    const vector<Matrix<int,2,2>>& pauli_str,
                    Mat& R_mat) const
    {
        /*
            compute the left matrix w.r.t site l for pauli string $alpha

            !!! Watch out the order of contracting sigma_i !!!
        */
        Mat buff(bond_dim, bond_dim);
        Mat buff_next(bond_dim, bond_dim);
        
        buff.setZero(); 
        buff_next.setZero();

        // num of row of B matrix 
        unsigned num_cols = mat_vec[0][site_num-1].rows(); 
        unsigned num_rows = num_cols; 

        unsigned num_cols_next;
        unsigned num_rows_next;
        
        if (pauli_str[site_num-1].isIdentity())
            buff.topLeftCorner(num_rows,num_cols).setIdentity(); 
        else 
        {
            for (int i=0; i<d; i++)
                for (int j=0; j<d; j++)
                {
                    if (pauli_str[site_num-1](i,j)!=0)
                        buff.topLeftCorner(num_rows, num_cols)   // order of i and j is important 
                            += pauli_str[site_num-1](i,j) * mat_vec[j][site_num-1] * mat_vec[i][site_num-1].adjoint();
                }         
        }
        
        for(int i = site_num-2; i > l ; i-- )
        {
            num_cols_next = mat_vec[0][i].rows();
            num_rows_next = num_cols_next;

            if (buff.topLeftCorner(num_rows,num_cols).isIdentity())
            {
                if (pauli_str[i].isIdentity())
                {
                    buff.topLeftCorner(num_rows_next,num_cols_next).setIdentity();
                }
                else
                {
                    buff_next.topLeftCorner(num_rows_next,num_cols_next).setZero();

                    for(int j=0; j<d; j++)
                        for(int k=0; k<d; k++)
                        {
                            if (pauli_str[i](j,k)!=0)
                                buff_next.topLeftCorner(num_rows_next,num_cols_next)
                                    += pauli_str[i](j,k) * mat_vec[k][i] * mat_vec[j][i].adjoint();
                        }
                    
                    buff.topLeftCorner(num_rows_next,num_cols_next)
                        =   buff_next.topLeftCorner(num_rows_next,num_cols_next); 
                }
            }   
            else 
            {
                /*
                    non-identity sandwiched term
                */
                buff_next.topLeftCorner(num_rows_next,num_cols_next).setZero();
                
                for(int j=0; j<d; j++)
                    for(int k=0; k<d; k++)
                    {
                        if (pauli_str[i](j,k)!=0) 
                            buff_next.topLeftCorner(num_rows_next,num_cols_next)
                                += pauli_str[i](j,k) * mat_vec[k][i] * buff.topLeftCorner(num_rows, num_cols) * mat_vec[j][i].adjoint();      

                    }
                buff.topLeftCorner(num_rows_next,num_cols_next)
                    =   buff_next.topLeftCorner(num_rows_next,num_cols_next); 
            }       

            num_rows = num_rows_next;
            num_cols = num_cols_next;
        }

        R_mat = buff.topLeftCorner(num_rows,num_cols);

    }

    void get_L_mat( 
                    unsigned l,
                    const vector<Matrix<int,2,2>>& pauli_str,
                    Mat& L_mat) const
                
    {   
        
        /*
            compute the left matrix w.r.t site l for pauli string $alpha
        
            Note :
            1. Corner cases: l=0 or l = L 
            2. L_mat must be resized before accessing . 
            3. Should enforce a small thread num by omp_set_thread_num() .etc 
        */

        // avoid dynamic allocation along the way
        Mat buff(bond_dim, bond_dim);
        Mat buff_next(bond_dim, bond_dim);
        
        buff.setZero(); 
        buff_next.setZero();
        
        unsigned num_col = mat_vec[0][0].cols();
        unsigned num_row = mat_vec[0][0].cols();

        unsigned num_col_next;
        unsigned num_row_next;
        
        // return identity when encountering W = I  
        if (pauli_str[0].isIdentity())
            buff.topLeftCorner(num_row,num_col).setIdentity();
        else 
        {
            for (int i=0; i<d; i++)
                for (int j=0; j<d; j++)
                {
                    if (pauli_str[0](i,j)!=0)
                        buff.topLeftCorner(num_row, num_col)+= pauli_str[0](i,j) * mat_vec[i][0].adjoint() * mat_vec[j][0];
                }         
        }

        for (int i=1; i<l; i++)
        {
            num_col_next = mat_vec[0][i].cols();
            num_row_next = num_col_next;

            if (buff.topLeftCorner(num_row, num_col).isIdentity())
            {
                /*
                    the sandwiched term is identity
                    take advantage of the canonicalization
                */
                if (pauli_str[i].isIdentity())
                {
                    buff.topLeftCorner(num_row_next,num_col_next).setIdentity();
                    //buff_next.topLeftCorner(num_row_next,num_col_next);
                } 
                else 
                {   
                    buff_next.topLeftCorner(num_row_next,num_col_next).setZero();

                    for(int j=0; j<d; j++)
                        for(int k=0; k<d; k++)
                        {
                            if (pauli_str[i](j,k)!=0)
                                buff_next.topLeftCorner(num_row_next,num_col_next)
                                    += pauli_str[i](j,k) * mat_vec[j][i].adjoint() * mat_vec[k][i];
                        }
                    // copy back. How to avoid copying things? 
                    buff.topLeftCorner(num_row_next,num_col_next)
                        =   buff_next.topLeftCorner(num_row_next,num_col_next); 
                }
            }
            else 
            {
                /*
                    non-identity sandwiched term
                */
                
                buff_next.topLeftCorner(num_row_next,num_col_next).setZero();

                for(int j=0; j<d; j++)
                    for(int k=0; k<d; k++)
                    {
                        if (pauli_str[i](j,k)!=0) 
                            buff_next.topLeftCorner(num_row_next,num_col_next)
                                += pauli_str[i](j,k) * mat_vec[j][i].adjoint() * buff.topLeftCorner(num_row, num_col) * mat_vec[k][i];      
                        
                    }
                
                buff.topLeftCorner(num_row_next,num_col_next)
                    =   buff_next.topLeftCorner(num_row_next,num_col_next); 
            }   
            
            num_row = num_row_next;
            num_col = num_col_next;
        }
        
        // finalize
        //L_mat.resize(num_row,num_col);
        L_mat = buff.topLeftCorner(num_row,num_col);
    }
    void multiply_scalar(double coeff)
    {
        mat_vec[0][0] *= coeff;
        mat_vec[1][0] *= coeff; 
    }

    void apply_single_h(const PauliStr& h, MPS& phi) 
    {
        for (int i=0; i< site_num; i++)
        {
            
            if (h[i].isIdentity()) 
            {
                for (int j=0; j<d; j++)
                {
                    phi.mat_vec[j][i] = mat_vec[j][i];
                }
            }
            else
            {
                for (int j=0; j<d; j++)
                {
                    Mat temp = Mat::Zero(mat_vec[0][i].rows(), mat_vec[0][i].cols());

                    for(int k=0; k<d; k++)
                    {
                        if (h[i](j,k)!=0)
                            temp = temp + h[i](j,k) * mat_vec[k][i]; 
                    }
                    phi.mat_vec[j][i] = temp;
                }
            }
        }
    }   

    void apply_single_h_inplace(const PauliStr& h)
    {
        /*
            apply a single pauli string by just re-combining the on-site matrices
        */
        
        for (int i=0; i< site_num; i++)
        {
            
            /*
            //if (1==0) 
            {
                for (int j=0; j<d; j++)
                {
                    phi.mat_vec[j][i] = mat_vec[j][i];
                }
            }   */

            if (!h[i].isIdentity()) 
            {
                vector<Mat> temp(d);
                for (auto& mat: temp)
                    mat = Mat::Zero(mat_vec[0][i].rows(), mat_vec[0][i].cols());
                
                for (int j=0; j<d; j++)
                {
                    //Mat temp = Mat::Zero(mat_vec[0][i].rows(), mat_vec[0][i].cols());

                    for(int k=0; k<d; k++)
                    {
                        if (h[i](j,k)!=0)
                            temp[k] = temp[k] + h[i](j,k) * mat_vec[k][i]; 
                    }
                    //phi.mat_vec[j][i] = temp;
                }

                for (int j=0; j<d; j++)
                    mat_vec[j][i] = temp[j];  
            }
        }
    }
};

#endif 