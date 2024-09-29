#define EIGEN_DONT_PARALLELIZE

#ifndef MPO_HPP
#define MPO_HPP

#include <Eigen/Eigen>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <complex>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <stdexcept>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <set>

//#include "mps.hpp"

using namespace Eigen;
using namespace std; 

template <typename T> class Diag_MPO
{
    /*
        the naive diagonal MPO  Dw ~= L^4 
        consists of pauli strings with Dw=2

        Keller et al. 's Dw ~= L^2 might be worth trying at some point
        (if we want to keep doing sweeping...)
        some matrix lazy evaluation? 
        
    */ 
    
    typedef Matrix<T,Dynamic,Dynamic> Mat;
    typedef Matrix<T,Dynamic,1> Vec;
    typedef Matrix<T,1,Dynamic> RowVec;       
    
    typedef Matrix<int,2,2> PauliMat;       // pauli matrices
    typedef vector<PauliMat> PauliStr; 
    typedef vector<Mat> MatOnsite; 

public:

    //unsigned bond_dim = 0; 
    unsigned num_bucket;   // workers spreaded based on alpha 
    unsigned site_num;
    unsigned d;            // local physical degree of freedom. For spin d=2    
    
    /*
        not differentiating nuclear term, 1b & 2b terms
    */

    vector<T> coeff;      // coefficient
    vector<PauliStr> h;   // Pauli string
    vector<string> label;   // for debug purpose 

    
    vector<T> coeff_pauli; // coeff in IXYZ representation
    vector<PauliStr> h_pauli; // mpo in IXYZ representation 
    vector<string> label_pauli; 
    double const_coeff_pauli; 

    vector<int> idx_2b, idx_1b;
    
    //adjustment for 1b integrals
    // t_ij = sum k [ik|kj]: be extremely careful about what to be included 
    Mat t_prime; 
    PauliMat Z, A_create, A_annihilate,n,X,Y,I;
    
    /// constructors
    Diag_MPO(){}

    Diag_MPO(   unsigned site_num_val,
                unsigned d_val,
                unsigned num_bckt_val,
                string filename,
                string scf_type)
    {
        //bond_dim = bond_dim_val;
        initialize(site_num_val,d_val,num_bckt_val,filename,scf_type);
    }   
    
    void initialize(unsigned site_num_val,
                    unsigned d_val,
                    unsigned num_bckt_val,  // number of thread to be used 
                    string filename,
                    string scf_type)
    {
        num_bucket = num_bckt_val;
        site_num = site_num_val;
        d = d_val;
        
        /// under the local basis adopted, should be -Z!
        /// very evil 
        Z << -1, 0, 0, 1;   

        A_create << 0, 1, 0, 0;
        A_annihilate << 0, 0, 1, 0;
        n << 1, 0, 0, 0; 
        X << 0, 1, 1 ,0;
        Y << 0,-1, 1 ,0; // not the real Pauli Sy 
        I << 1, 0, 0, 1; 
        
        t_prime = Mat::Zero(site_num,site_num);
        
        // extra coeff of identity string in IXYZ rep. 
        const_coeff_pauli = 0;

        if (scf_type!="RHF")
            throw invalid_argument("Only support RHF integrals");   
        
        fill_from_fcidump(filename,scf_type);
    }

    void set_identity_str(PauliStr& tgt)
    {
        tgt.resize(site_num);
        for (int i=0; i<site_num; i++)
            tgt[i] = PauliMat::Identity();       
    }

    void take_hermitian(PauliStr& tgt, unsigned i)
    {
        tgt[i].adjointInPlace(); 
    }
    void set_Z(PauliStr& tgt, unsigned start, unsigned end)
    {
        /*
            [start+1, end-1]
        */
        for(int i=start+1; i < end; i++)
        {
            tgt[i] = Z;
        }
    }

    void push_back_coeff_n_str(double coeff_val, PauliStr pauli_str)
    {
        coeff.push_back(coeff_val);
        h.push_back(pauli_str);
    }
    
    void fill_from_fcidump(string filename, string scf_type)
    {
        /*
            Suppport RHF FCIDUMP generated from psi4/PySCF
            Old-fashioned C style reading...
        */
        bool start_read_int = false;

        int num_lines;
        double coeff_tmp;
        int i,j,k,l;
        
        const char* u = &filename[0];

        if (scf_type=="RHF")
        {   
            FILE *fp = fopen(u,"r");
            fscanf(fp,"%d",&num_lines);

            cout << endl << "***LOADING INTEGRALS...***" <<endl; 
            cout << "number of FCIDUMP lines: " << num_lines <<endl; 
            
            for (int line = 0 ; line < num_lines; line++)
            {
                fscanf(fp, "%lf %d %d %d %d", &coeff_tmp, &i, &j, &k, &l);
                //cout << coeff <<endl;
                to_pauli_str(coeff_tmp,i,j,k,l);

                //cout << coeff <<endl; 
            }
            
            fclose(fp);
            
            /*
            for (auto num:coeff)
            {
                cout << num << endl; 
            }*/

            cout << "number of Pauli strings: " << h.size() <<endl;  
            cout << "***INTEGRALS LOADED***" <<endl<<endl;;
            //cout << t_prime <<endl;
        }
        else 
        {   
            throw invalid_argument("Only support RHF integrals");   
        }
    }
    
    bool is_identity(PauliStr& q)
    {
        for (auto w: q)
            if (w != I) return false;
        
        return true; 
    }

    int get_term_order(string label)
    {   
        // determine 1b or 2b 
        stringstream q(label);
        string temp; 
        char delim = ' ';
        int num= 0 ; 

        while(std::getline(q,temp,delim))
        {
            num ++;   
        }

        return num;
    }

    void compress_IXYZ(int start_idx, int end_idx, string label)
    {
        /*
            tranform to IXYZ representation 
        */

        // handle the label 
        string temp;
        stringstream q(label);
        char delim = ' ';
        set<int> qq; 
        vector<int> op_site_idx; 

        unordered_map<string,double> hash;  

        while (std::getline(q,temp,delim))
        {
            qq.insert(stoi(temp)); 
        }

        for (auto ite = qq.begin(); ite != qq.end(); ite++)
            op_site_idx.push_back(*ite);

        for (int i=start_idx; i< end_idx; i++ )
        {   

            for (int t = 0; t < pow(2,op_site_idx.size()) ; t++) 
            {
                // 2^N binary strings 
                
                string key = "";
                double local_coeff = coeff[i] * pow(0.5, op_site_idx.size()); 

                for (int s = 0; s < op_site_idx.size(); s++)
                {
                    int base = pow(2,s);
                    int k = t & base; // 0 or base 

                    int idx = op_site_idx[s]; 

                    if (h[i][idx] == n)
                    {
                        // n = 1/2 (I - Z)
                        if (k==0 )
                        {
                            key += "I";
                        }
                        else
                        {
                            key += "Z";
                            local_coeff *= -1;
                        }
                    }
                    else if (h[i][idx] == n+Z)
                    {
                        // n+Z = 1/2 (I + Z)
                        if (k==0)
                        {
                            key += "I";
                        }
                        else 
                        {
                            key += "Z";
                        }
                    }
                    else if (h[i][idx] == A_create)
                    {
                        // A_create = 1/2 (X-Y)
                        if (k==0)
                        {
                            key += "X";
                        }   
                        else 
                        {
                            key += "Y";
                            local_coeff *= -1;
                        }
                    }
                    else if ((h[i][idx] == A_annihilate))
                    {
                        //A_annhilate = 1/2 (X+Y)
                        if(k==0)
                        {
                            key += "X";
                        }
                        else
                        {
                            key += "Y";
                        }
                    }
                    else
                    {
                        cout << "something is wrong\n";
                        cout << h[i][idx] << endl;
                    }
                }   
                
                //cout << local_coeff << "," << key <<endl; 
                if (hash.find(key) == hash.end())
                    hash[key] = local_coeff; 
                else 
                    hash[key] += local_coeff;                     
            }
            
        }
        
        for (auto ite = hash.begin(); ite != hash.end(); ite++)
            if (abs(ite->second) > 1e-15)
            {
                //cout << ite->first << ":" << ite->second <<endl; 
                
                auto temp = h[start_idx];
                auto label_now  = ite->first ;

                // collect const_coeff_pauli 

                for (int i=0; i< op_site_idx.size();i++ )
                {
                    if (label_now[i] == 'I')
                    {
                        temp[op_site_idx[i]] = I;
                    }
                    else if (label_now[i] == 'X')
                    {
                        temp[op_site_idx[i]] = X;
                    }
                    else if (label_now[i] == 'Y')
                    {
                        temp[op_site_idx[i]] = Y;
                    }
                    else if (label_now[i] == 'Z')
                    {
                        temp[op_site_idx[i]] = Z;
                    }
                    else
                    {   
                        
                        //cout << "somehting went wrong\n";
                        cout << label_now <<endl; 
                        throw invalid_argument("operator symbol other than I, X, Y, or Z was found");
                    }
                    
                } 
                
                if (is_identity(temp))
                //if (0==1)
                {
                    // identity string 
                    // wrong 
                    const_coeff_pauli += ite->second; 

                    //cout << "identity string found in " << label <<endl;
                    
                }
                else 
                {
                    h_pauli.push_back(temp);
                    coeff_pauli.push_back(ite->second);
                    label_pauli.push_back(label_now);
                }
            }
            
    }
    void transform_to_pauli()
    {
        /*
            Re-write MPO in terms of "Pauli" matrices 
            Y = 0 -1 
                1  0
        */
        int start_idx = 0 ;
        int end_idx = 0; 
        string start_label = label[0];
        
        for (int i=1; i<h.size(); i++)
        {   
            /*
            if (label[i] == "24 24")
            {
                //cout << coeff[i] <<endl; 
                compress_IXYZ(i,i+1,label[i]);
                break; 
            }*/
            
            if (label[i] != start_label)
            {
                end_idx = i; 

                //cout << start_idx << "," <<end_idx << ":" << start_label <<endl;  
                // compress 
                compress_IXYZ(start_idx,end_idx,start_label); 

                start_idx = i ;
                start_label = label[i]; 
            }
            
        }
        // tail is just the identity string 
        
        coeff_pauli.push_back(coeff[h.size()-1] + const_coeff_pauli);
        h_pauli.push_back(h[h.size()-1]);
        label_pauli.push_back("const");

        cout << "Num of terms in IXYZ rep: " << h_pauli.size() <<endl; 
        

    }
    
    void enum_sym_term_ijkl(T coeff_val,
                            int i,
                            int j,
                            int k,
                            int l)
    {
        /*
            given an intergal w. i j k l, 
            enumerate all symmetric terms and pushback to self.h
            also calculate t_prime for one body term 
            
            i j k l are spin orbital indices
            which still obeys the relation of spatial orbs

            Note:
            Only considered real coeff because I'm lazy
        */
        PauliStr temp; 
        //set_identity_str(temp);  

        string label_tmp = to_string(i)+ " " 
                              +to_string(j)+ " "
                              +to_string(k)+ " "
                              +to_string(l);

        if (i==j && j==k && k==l)
        {   
            // CLEAR
            //[ii|ii]
            set_identity_str(temp);
            temp[i] = n;
            
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            
            /*
            cout << label_tmp <<endl;
            cout << t_prime(2,2) <<endl; 
            cout << coeff_val <<endl; 
            cout << endl;
            */
            // sum [ik|kj]
            t_prime(i,l) += coeff_val;
            /*
            cout <<"in ["<<i<<j<<"|"<<k<<l<<"]\n" ;
            cout << coeff_val <<endl;
            cout << t_prime <<endl;
            */
            label.push_back(label_tmp);
            
        }
        else if(i==j && j==k && k!=l)
        {
            // [ii|ij]
            // i > j 
            // CLEAR
            // 3 3 3 1
            // i j k l 
            set_identity_str(temp);
            
            int j_real = l ;

            set_Z(temp,j_real,i);
            temp[j_real] = A_annihilate;
            temp[i] = A_create;
            
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);      
            label.push_back(label_tmp);

            // h.c. 
            coeff.push_back(0.5*coeff_val);
            take_hermitian(temp,i);
            take_hermitian(temp,j_real);
            //temp[j] = temp[j].adjoint();
            //temp[i] = temp[i].adjoint(); 
            h.push_back(temp);
            label.push_back(label_tmp);

            t_prime(i,l) += coeff_val;
            t_prime(l,i) += coeff_val; 


        }   
        else if (i!=j && j==k && k==l)
        {
            // CLEAR
            // [ij|jj], i>j 
            // 3 1 1 1 
            // i j k l 
            set_identity_str(temp);
            set_Z(temp,j,i);
            
            temp[j] = A_create;
            temp[i] = A_annihilate;
            
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);

            coeff.push_back(0.5*coeff_val);
            take_hermitian(temp,i);
            take_hermitian(temp,j);
            h.push_back(temp);
            label.push_back(label_tmp);

            t_prime(i,j) += coeff_val;
            t_prime(j,i) += coeff_val;

        }   
        else if (i==k && i!=j && j==l)
        {
            // raw: [ij|ij] i>j 

            //[ij|ji]
            
            set_identity_str(temp);
            temp[j] = n+Z;
            temp[i] = n;
            
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            //[ji|ij]
            temp[j] = n;
            temp[i] = n+Z;
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            t_prime(i,i) += coeff_val;
            t_prime(j,j) += coeff_val;
            
        }
        else if (i==j && j!=k && k==l)
        {
            //[ii|jj] i>j
            // CLEAR 
            set_identity_str(temp);
            temp[i] = n;
            temp[k] = n;
            
            coeff.push_back(coeff_val);
            h.push_back(temp);    
            label.push_back(label_tmp);
      
        }
        else if (i==k && i!=j && k!=l && j!=l)
        {
            //[ij|ik] i>j & i>k
            // 5 3 5 1 
            // i j k l 
            set_identity_str(temp); 
            
            int j_small = min(j,l);
            int k_big = max(j,l);
            // term 1
            set_Z(temp,j_small,k_big);
            temp[j_small] = A_create;
            temp[k_big] = A_annihilate;
            temp[i] = Z + n; 
            
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp); 
            label.push_back(label_tmp);
            
            // h.c. 
            coeff.push_back(0.5*coeff_val);
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            h.push_back(temp); 
            label.push_back(label_tmp);

            //term 2 
            temp[j_small] = A_create;
            temp[k_big] =  A_annihilate;
            temp[i] = n; 
            coeff.push_back(-0.5*coeff_val);
            h.push_back(temp); 
            label.push_back(label_tmp);

            coeff.push_back(-0.5*coeff_val);
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            h.push_back(temp); 
            label.push_back(label_tmp);

            t_prime(j,l) += coeff_val;
            t_prime(l,j) += coeff_val;
            
        }
        else if(j==l && i!=j && k!=l && i!=k)
        {
            //[ji|ki] j>i, k>i
            // effectively ~ [ij|ik] , i< j & k 
            
            // 4 2 3 2 
            // i j k l 
            set_identity_str(temp); 
            
            int i_real = j; 
            int j_small = min(i,k);
            int k_big = max(i,k);

            set_Z(temp, j_small, k_big);

            temp[i_real] = n + Z;
            temp[j_small] = A_create; 
            temp[k_big] = A_annihilate;

            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);

            // h.c. 
            coeff.push_back(0.5*coeff_val);
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            // another term 
            temp[i_real] = n;
            temp[j_small] = A_create;
            temp[k_big] = A_annihilate;
            coeff.push_back(-0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            // h.c.
            coeff.push_back(-0.5*coeff_val);
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            t_prime(i,k) += coeff_val;
            t_prime(k,i) += coeff_val;  
            
        }
        else if (i!=j && j==k && k!=l)
        {
            // should be CLEAR. Or diagonal elements'll be wrong 
            // actual term: [ij|jk] i> j > k 
            // effectively case "[ij|ik], j < i < k" in appendix
            // 4 3 3 2
            // i j k l     
            set_identity_str(temp);
            
            // j_small < i < k_big 
            int i_real = j;
            int j_small = min(i,l);
            int k_big = max(i,l);

            // JW
            set_Z(temp,j_small, i_real);
            set_Z(temp,i_real, k_big);

            // 1
            temp[j_small] = A_create ;
            temp[i_real] = n;
            temp[k_big] = A_annihilate;
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);

            // h.c. 
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);

            // 2
            temp[j_small] = A_create * Z;
            temp[i_real] = Z + n; 
            temp[k_big] = A_annihilate;
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);

            // h.c. 
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            coeff.push_back(0.5*coeff_val);
            h.push_back(temp);
            label.push_back(label_tmp);
            
            t_prime(i,l) += coeff_val;
            t_prime(l,i) += coeff_val;

        }
        else if(i==j && j!=k && k!=l)
        {
            // CLEAR 0309
            // [ii|jk] i > j & k 
            // case j < k < i in the appendix  
            set_identity_str(temp);

            // 3 3 2 0 
            // i j k l 
            int j_small = min(k,l);
            int k_big = max(k,l);

            set_Z(temp,j_small,k_big);

            temp[j_small] = A_create;
            temp[k_big] = A_annihilate;
            temp[i] = n;
            coeff.push_back(coeff_val);    // 2 in pre-factor 
            h.push_back(temp);
            label.push_back(label_tmp);

            // h.c. 
            coeff.push_back(coeff_val);
            take_hermitian(temp,j_small);
            take_hermitian(temp,k_big);
            h.push_back(temp);
            label.push_back(label_tmp);


        }
        else if(i!=j && j!=k && k==l )
        {
            // CLEAR
            //[ij|kk] , k < i & j 
            // effectively ~ [kk|ij] k < i & j, case "i<j<k" in the appendix
            
            /*
                somthing is wrong here 
                k should have the particle number operator 
            */
            
            set_identity_str(temp);

            if (k<j)
            {
                /*
                    case "i < j < k" 
                */
                int i_real = k; 
                int j_real = min(i,j);
                int k_real = max(i,j);
                
                // term 1
                set_Z(temp,j_real,k_real);
                temp[i_real] = n;
                temp[j_real] = A_create;
                temp[k_real] = A_annihilate;
                coeff.push_back(coeff_val);    // 2 in pre-factor 
                h.push_back(temp);
                label.push_back(label_tmp);

                //h.c.
                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                coeff.push_back(coeff_val);    // 2 in pre-factor 
                h.push_back(temp);
                label.push_back(label_tmp);
            }
            else 
            {
                /*
                    case "j < i < k"
                */
                
                int j_real = min(i,j);
                int i_real = k; 
                int k_real = max(i,j); 
                
                set_Z(temp,j_real,i_real);
                set_Z(temp,i_real,k_real);

                temp[j_real] = A_create;
                temp[i_real] = n;
                temp[k_real] = A_annihilate;

                coeff.push_back(-coeff_val);    // 2 in pre-factor 
                h.push_back(temp);
                label.push_back(label_tmp);

                // h.c. 

                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                coeff.push_back(-coeff_val);    // 2 in pre-factor 
                h.push_back(temp);
                label.push_back(label_tmp);
            }

        }
        else if (i!=j && j!=k && k!=l && i!=l)
        {
            // SHOULD BE CLEAR
            /*
                [ij|kl] : i>j, k>l; i>k
            */

            // ensure the smallest appears on the left 
            if (l<j)
            {
                swap(i,k);
                swap(j,l);
            }
            // ensure i < j, so i is the smallest
            swap(i,j);

            int k_small = min(k,l);
            int l_big = max(k,l); 

            /// three cases
            if (j < l_big && j < k_small)
            {
                // case "[ij|kl], i< j< k < l" in the appendix 
                set_identity_str(temp); 
                
                set_Z(temp,i,j);
                set_Z(temp,k_small, l_big);

                // +-+- 
                temp[i] = A_create; 
                temp[j] = A_annihilate;
                temp[k_small] = A_create;
                temp[l_big] = A_annihilate;

                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);
                
                //h.c.
                take_hermitian(temp,i);
                take_hermitian(temp,j);
                take_hermitian(temp,k_small);
                take_hermitian(temp,l_big);
                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);
                
                // -++-
                temp[i] = A_annihilate;
                temp[j] = A_create;
                temp[k_small] = A_create;
                temp[l_big] = A_annihilate; 
                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                // h.c. 
                take_hermitian(temp,i);
                take_hermitian(temp,j);
                take_hermitian(temp,k_small);
                take_hermitian(temp,l_big);
                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

            }
            else if (j > k_small && j < l_big) 
            {
                // case "[ik|jl], i< j< k < l" in the appendix 
                // real order i < k_small < j < l_big
                set_identity_str(temp);
                
                int i_real = i;
                int j_real = k_small;
                int k_real = j;
                int l_real = l_big;

                set_Z(temp,i_real,j_real);
                set_Z(temp,k_real,l_real);
                
                // ++--
                temp[i_real] = A_create;
                temp[j_real] = A_create;
                temp[k_real] = A_annihilate;
                temp[l_real] = A_annihilate;
                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                // h.c. 
                take_hermitian(temp,i_real);
                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                take_hermitian(temp,l_real);
                coeff.push_back(coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                // -++-
                temp[i_real] = A_annihilate;
                temp[j_real] = A_create;
                temp[k_real] = A_create;
                temp[l_real] = A_annihilate;
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);
                
                //h.c. 
                take_hermitian(temp,i_real);
                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                take_hermitian(temp,l_real);
                
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);
                
                                
            }             
            else if (j > l_big)
            {
                // case "[il|jk], i< j< k < l" in the appendix 
                // real order : i < k_small < l_big < j 
                set_identity_str(temp);

                // JW
                int i_real = i;
                int j_real = k_small;
                int k_real = l_big;
                int l_real = j;
                
                set_Z(temp, i_real, j_real);
                set_Z(temp, k_real, l_real);

                // ++--
                temp[i_real] = A_create;
                temp[j_real] = A_create;
                temp[k_real] = A_annihilate;
                temp[l_real] = A_annihilate;
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                // h.c. 
                take_hermitian(temp,i_real);
                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                take_hermitian(temp,l_real);
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                //-+-+
                temp[i] = A_annihilate;
                temp[k_small] = A_create;
                temp[l_big] = A_annihilate;
                temp[j] = A_create;
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);

                //h.c.
                take_hermitian(temp,i_real);
                take_hermitian(temp,j_real);
                take_hermitian(temp,k_real);
                take_hermitian(temp,l_real);
                coeff.push_back(-coeff_val);
                h.push_back(temp);
                label.push_back(label_tmp);
            }   
        }
        else 
        {
            // shouldn't reach this step if things are all correct
            cout << "something is missing:" << i<<" "<<j <<" "<<k <<" "<<l <<endl;
        }
    

    }
    void to_pauli_str(  T coeff_val,
                        int i,
                        int j,
                        int k,
                        int l)
    {
        /*
            PURGATORY STARTS HERE
            
            expand symmetric terms based the "clean" FCIDUMP 
            
            For [ij|kl]:
            if i != k 
                i>=j, k>=l, i>=k
            if i == k
                i>=j, k>=l, i>=k, j>= l
                
            2b term: uu, dd, ud
            1b term: u d 
            
            !!!Also: collect sum_k [ik|kj] for 1b terms

            Spatial orb to spin orb

            i, j, k ,l ~ [1,Norb]
            i_spin ~ [0,2*Norb-1]
            i*2 - 2 ~ (|i,d>)
            i*2 - 1 ~ (|i,u>)
        */ 
        int num_2b=0, num_1b=0; 
        
        if ( i!=0 && j!=0 && k!=0 && l!=0)
        {
            // two-body term
            //PauliStr temp; 
            //set_identity_str(temp);  
            
            // |0>|0>
            //return;  
            int i_spin = 2*i - 1; 
            int j_spin = 2*j - 1;  
            int k_spin = 2*k - 1;
            int l_spin = 2*l - 1;

            //cout << i_spin << " " << j_spin << " "<< k_spin <<" " << l_spin <<endl;
            //double temp = t_prime(2,2);
            enum_sym_term_ijkl(coeff_val, i_spin, j_spin, k_spin, l_spin);
            /*
            if (temp!=t_prime(2,2))
            {
                cout << coeff_val << endl;
                cout << i_spin << " " << j_spin << " "<< k_spin <<" " << l_spin <<endl;
            }*/
            //cout <<"after:" << t_prime(2,2) <<endl;

            // |1>|1>
            i_spin = 2*i - 2; 
            j_spin = 2*j - 2;  
            k_spin = 2*k - 2;
            l_spin = 2*l - 2;

            //temp = t_prime(2,2);
            enum_sym_term_ijkl(coeff_val, i_spin, j_spin, k_spin, l_spin);
            /*
            if (temp!=t_prime(2,2))
            {
                cout << coeff_val << endl;
                cout << i_spin << " " << j_spin << " "<< k_spin <<" " << l_spin <<endl;
            }*/
            
            // |0>|1>
            i_spin = 2*i - 1; 
            j_spin = 2*j - 1;  
            k_spin = 2*k - 2;
            l_spin = 2*l - 2;

            //temp = t_prime(2,2);
            enum_sym_term_ijkl(coeff_val, i_spin, j_spin, k_spin, l_spin);
            /*
            if (temp!=t_prime(2,2))
            {
                cout << coeff_val << endl;
                cout << i_spin << " " << j_spin << " "<< k_spin <<" " << l_spin <<endl;
            }*/
        
            // |1>|0>
            // in this case the ordering can be broken. Need adjustments
            //  
            if ((!(i==j && j==k && k==l)) && (!(i==k && j!= k && j==l )) )
            {
                i_spin = 2*i - 2; 
                j_spin = 2*j - 2;  
                k_spin = 2*k - 1;
                l_spin = 2*l - 1;

                if (k_spin > i_spin)
                {
                    swap(i_spin,k_spin);
                    swap(j_spin,l_spin);
                }
                
                enum_sym_term_ijkl(coeff_val, i_spin, j_spin, k_spin, l_spin);
            }
            
        }
        //num_2b = coeff.size(); 
        //cout << "number of 2b terms: " << num_2b <<endl; 

        if ( i!=0 && j!=0 && k==0 && l==0)
        {
            // one-body term
            /*
                i > j !
            */
            PauliStr temp; 

            //cout << i<<" "<<j <<" "<<k <<" "<<l <<endl;
            //if (!(i==1&&j==1)) return;
            //u
            int i_spin = 2*i - 1; 
            int j_spin = 2*j - 1;  

            string label_tmp = to_string(i_spin)+ " " 
                              +to_string(j_spin);

            if (i_spin == j_spin)
            {
                set_identity_str(temp); 
                temp[i_spin] = n;

                coeff.push_back(coeff_val - 0.5*t_prime(i_spin,j_spin));
                h.push_back(temp);
                label.push_back(label_tmp);
            } 
            else
            {   
                set_identity_str(temp);
                set_Z(temp,j_spin, i_spin);

                //i j 
                temp[i_spin] = A_create;
                temp[j_spin] = Z * A_annihilate;

                coeff.push_back(coeff_val - 0.5*t_prime(i_spin, j_spin));
                coeff.push_back(coeff_val - 0.5*t_prime(i_spin, j_spin));


                h.push_back(temp);
                label.push_back(label_tmp);

                temp[i_spin].adjointInPlace();
                temp[j_spin].adjointInPlace(); 
                h.push_back(temp);
                label.push_back(label_tmp);
            }
            //return;
            //d
            
            i_spin = 2*i - 2;
            j_spin = 2*j - 2;

            label_tmp = to_string(i_spin)+ " " 
                              +to_string(j_spin);

            if (i_spin == j_spin)
            {
                set_identity_str(temp);
                temp[i_spin] = n;

                coeff.push_back(coeff_val - 0.5*t_prime(i_spin,j_spin));
                h.push_back(temp);
                label.push_back(label_tmp);
            } 
            else
            {   
                // i_spin > j_spin 
                set_identity_str(temp);
                set_Z(temp,j_spin,i_spin);

                //i j 
                temp[i_spin] = A_create * Z;
                temp[j_spin] = A_annihilate;
                
                coeff.push_back(coeff_val - 0.5*t_prime(i_spin, j_spin));
                coeff.push_back(coeff_val - 0.5*t_prime(i_spin, j_spin));

                h.push_back(temp);
                temp[i_spin].adjointInPlace();
                temp[j_spin].adjointInPlace(); 
                h.push_back(temp);

                label.push_back(label_tmp);
                label.push_back(label_tmp);

            }
        }

        //cout << "number of 1b terms: " << coeff.size() - num_2b <<endl; 
        
        if (i==0 && j==0 && k==0 && l==0)
        {
            // nuclear term 
            string label_tmp = "const";

            coeff.push_back(coeff_val);
            PauliStr temp; 
            set_identity_str(temp);    
            h.push_back(temp);
            label.push_back(label_tmp);
            
        }
        

    }

    void apply_mps( const MPS<T>& psi, 
                    const vector<Mat>& M_input,  // d on-site matrices as input     
                    vector<Mat>& Q_final,  // d resultant matrices
                    unsigned site_l) 
    {   
        /*
            applying H on mixed canonical psi w.r.t the local basis on site l 
            parallelized with openmp, but can be easily rewritten in MPI
            Note
            1. save N matrices and avoid computation. Just a few of possibilities

        */
        unsigned num_row = M_input[0].rows();
        unsigned num_col = M_input[0].cols();
        
        // bucket for each thread 
        vector<Mat> L_mat(num_bucket);
        vector<Mat> R_mat(num_bucket);
        vector<vector<Mat>> Q(d);
        
        for(int i=0; i< d; i++)
            Q[i].resize(num_bucket);
        
        for(int i=0; i< num_bucket; i++)
        {
            L_mat[i] = Mat::Zero(num_row,num_row);  
            R_mat[i] = Mat::Zero(num_col,num_col);
            
            for(int j=0; j< d; j++)
            {
                Q[j][i] = Mat::Zero(num_row,num_col); 
            }
        }
        //cout << "beep\n";
        //auto t1 = std::chrono::system_clock::now(); 
        double t1 = omp_get_wtime();

        #pragma omp parallel for num_threads(num_bucket) schedule(dynamic, 50)
        for (unsigned alpha = 0; alpha < h.size(); alpha ++ )
        {
            /*
                Corner cases: site_l = 0 o site_l = num_site
                ? L and R here can be re-used with in a single Lanczos process
            */
            
            Mat temp(num_row,num_col);
            
            /// make sure omp works all right
            /*
            if (alpha==0)
            {
                cout << "num of thread for MPO MPS multiplication:" << omp_get_num_threads() <<endl ;
                cout << "num of thread used by Eigen:" << nbThreads() <<endl; 
            }
            */
            // determine bucket idx
            unsigned thread_idx = omp_get_thread_num();
            
            if (site_l > 0)
                psi.get_L_mat(site_l,h[alpha],L_mat[thread_idx]);
            
            if (site_l < site_num - 1)
                psi.get_R_mat(site_l,h[alpha],R_mat[thread_idx]); 
            
            for (unsigned i=0; i<d; i++)
            {
                temp.setZero();
                
                for (unsigned j=0; j<d; j++)
                    if (h[alpha][site_l](i,j)!=0)
                        //temp += h[site_l](i,j) * psi.mat_vev[j][site_l];
                        temp += h[alpha][site_l](i,j) * M_input[j];
                            
                // Q matrix of this paulis str
                if (site_l == 0 )
                    Q[i][thread_idx] += coeff[alpha] * temp * R_mat[thread_idx]; 
                else if (site_l == site_num-1)
                    Q[i][thread_idx] += coeff[alpha] * L_mat[thread_idx] * temp; 
                else 
                    Q[i][thread_idx] += coeff[alpha] * L_mat[thread_idx] * temp * R_mat[thread_idx]; 
            }        
        }
        //auto t2 = std::chrono::system_clock::now(); 
        double t2 = omp_get_wtime();

        //std::chrono::duration<double> t_elapsed = t2-t1;
        std::cout << "MPO-MPS multiplication execution time:" << t2-t1 <<endl<<endl;

        // reduction of Q matrix 
        Q_final.resize(psi.d);
        for (unsigned i = 0; i < psi.d; i++)
        {
            Q_final[i] = Mat::Zero(num_row,num_col);
            for(unsigned j=0; j< num_bucket; j++)
                Q_final[i] += Q[i][j];

            //cout << Q_final[i] <<"\n\n";
        }
    }   
    
    double calc_expectation(MPS<T>& psi)
    {
        /*
            calculate the expectation value of a single MPS that is not in canonical form
        */

        double sum = 0; 
           
        MPS<T> temp(psi.bond_dim,psi.site_num,psi.d); 

        for (int i=0; i<h.size(); i++ )
        {
            psi.apply_single_h(h[i],temp);
            sum += coeff[i] * psi.inner_product(temp);
        } 

        sum /= psi.get_norm();

        return sum; 

    }

    void to_1b_2b()
    {
        cout << "counting 1b and 2b operators..." <<endl; 
        for (int i=0; i< h.size(); i++)
        {
            //cout << label[i] << ", " << get_term_order(label[i]) <<endl;
            if (get_term_order(label[i]) > 2 )
            {
                idx_2b.push_back(i);
                //h_2b.push_back(h[i]);
            }
            else 
            {
                idx_1b.push_back(i);
            }
        }

        cout << "num of 1b and nuclear term: " << idx_1b.size() <<endl; 
        cout << "num of 2b terms: " << idx_2b.size() <<endl; 

    }
};
#endif