#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "sampler.hpp"

bool comp(pair<int,double> a, pair<int,double> b)
{
    return abs(a.second) > abs(b.second); 
}

double sampler::make_group()
{   
    /*
        sum_k (b) / sum_all (b) 
        ?
        Use mpo_pool.h_coeff 
    */     
    vector<pair<int,double>> idx_and_coeff; 
    
    // initialize group object 
    int num_group = mpo_pool.coeff.size() /max_group_size +1;

    //group.range.resize(num_group+1); 

    // 1st order approx of ITE!
    ite_op.idx_op.resize(num_group); 
    ite_op.range.push_back(0.0);
    idx_and_coeff.resize(mpo_pool.coeff.size());
    
    for(int i=0; i< mpo_pool.coeff.size();i++)
    {
        idx_and_coeff[i] = make_pair(i,-d_tau*mpo_pool.coeff[i]);
    }   
    
    // nuclear term is always at last 
    //idx_and_coeff[mpo_pool.coeff_pauli.size()-1] = make_pair(mpo_pool.coeff_pauli.size()-1, 1.0 - d_tau*mpo_pool.coeff_pauli[mpo_pool.coeff_pauli.size()-1]);
    
    // sort w.r.t abs(coeff) in descending order 
    sort(idx_and_coeff.begin(), idx_and_coeff.end(), comp);
    /*
    for (auto item:idx_and_coeff)
        cout <<item.first << " " << item.second << endl; 
    */
    /// grouping starts 
    /*
        use area under curve as the group weight
        forget about I for now 
    */
    
    double total_weight = 0; 
    int group_cnt = 0; 
    
    
    for (int i=0; i< idx_and_coeff.size(); i+= max_group_size)
    {
        double weight = 0; 
        
        int end = i+ max_group_size < idx_and_coeff.size() ?  i+ max_group_size : idx_and_coeff.size();

        for (int j=i; j < end ; j++)
        {
            ite_op.idx_op[group_cnt].push_back(idx_and_coeff[j]);
            
            weight += abs(idx_and_coeff[j].second);             
        }
        
        total_weight += weight; 
        
        cout << "weight of group " << group_cnt << ":"  << weight <<endl;  

        ite_op.range.push_back(total_weight);

        group_cnt++; 
        /*   
        if (group_cnt==1)
        {
            total_weight += (weight + weight*identity_scale );  
            cout << "weight of group " << group_cnt-1 << ":"  << identity_scale* weight <<endl; 
            cout << "weight of group " << group_cnt << ":"  << weight <<endl; 
            ite_op.range.push_back(identity_scale* weight);
            ite_op.range.push_back(total_weight);
        }
        else 
        {
            total_weight += weight; 
            cout << "weight of group " << group_cnt << ":"  << weight <<endl; 
            ite_op.range.push_back(total_weight);
        }*/
        //cout << total_weight <<endl; 
        //cout << i <<endl;
        
    }   
    //return; 

    // normalize 
    for (auto& w: ite_op.range)
    {
        w/=total_weight; 
        cout << w <<endl; 
    }  
      
    for (auto group : ite_op.idx_op)
        cout << "size: " << group.size() <<endl; 
    /*
    for (auto q: ite_op.idx_op[0])
        cout << q.first << "," << q.second<<  "," << mpo_pool.label[q.first] <<endl;
    */
}   

void sampler::run_test() 
{
    /*
        randomly chosen h   
        
    */   
    int num_h = mpo_pool.h.size();
    
    // pool for sampling
    vector<int> idx_list;

    for (int i=0; i< num_h; i++)
    {
        idx_list.push_back(i);
    }

    int num_thread = mpo_pool.num_bucket;  
    
    std::mt19937_64 random_engine(std::chrono::system_clock::now().time_since_epoch().count());

    uniform_int_distribution<int> dice(0,num_h-1);
    
    // random engine for each walker
    vector<mt19937_64> rand_engines(num_walker);

    // and distinct seeding using the state of device
    for (int i=0; i< num_walker; i++)
        rand_engines[i] = mt19937_64(std::random_device{}());
    
    cout << "num of deterministic operators: " << mpo_pool.idx_1b.size() <<endl;

    for (int step = 0; step < num_step;  step++ )
    {   
        /*
            WARNING! 
            DANGEROUS HARD-WIRING
        */
        /*
        if (step < 5000 && step > 1000) 
            d_tau = 0.02;
        if (step >= 5000) 
            d_tau = 0.01; 
        */
        // should be parallelized 

        #pragma omp parallel for num_threads(num_thread)
        for (int i=0; i< num_walker; i++)
        {   
            // deterministically 
            
            // controlled by is_semi. 
            // is_semi=false -> empty idx_1b
            
            //vector<int> idx_sampled(mpo_pool.idx_1b); 
            vector<int> idx_sampled; 
            

            // another hyper parameter! 
            int num_1b = 0 * num_rand_h;
            int num_2b = num_rand_h - num_1b; 

            if (is_semi==true)
            {
                // 1b
                std::sample(mpo_pool.idx_1b.begin(), 
                            mpo_pool.idx_1b.end(), 
                            std::back_inserter(idx_sampled),
                            num_1b,
                            rand_engines[i]); 

                // 2b
                std::sample(mpo_pool.idx_2b.begin(), 
                            mpo_pool.idx_2b.end(), 
                            std::back_inserter(idx_sampled),
                            num_2b,
                            rand_engines[i]); 
                
            }
            else
            {
                std::sample(idx_list.begin(), idx_list.end(),std::back_inserter(idx_sampled),num_rand_h,rand_engines[i]);         
            }
            // temp MPS
            MPS<double> result(walker[0].bond_dim,walker[0].site_num,walker[0].d);
            
            MPS<double> walker_copy; 
            walker_copy.blank_copy(walker[i]);      

            // 1 + d_tau* E
            //walker_copy.multiply_scalar(1+d_tau*E_step);
            
            // I - d_tau * sum_(i in G) (alpha_i h_i)
            for (int j = 0; j < idx_sampled.size(); j++) 
            {
                walker_copy.apply_single_h(mpo_pool.h[idx_sampled[j]],result);
                result.multiply_scalar(-d_tau * mpo_pool.coeff[idx_sampled[j]]); 
                
                MPS<double> sum; 
                sum.from_direct_sum(walker[i],result);
                sum.compress_svd(walker[i].bond_dim);
                
                walker[i].careless_copy(sum);
            }
            /*
            int chosen_idx = dice(gen);
            
            MPS<double> res_walker(walker[i].bond_dim, walker[i].site_num, walker[i].d); 
            MPS<double> direct_sum; 

            walker[i].apply_single_h(mpo_pool.h[chosen_idx],res_walker);
            res_walker.multiply_scalar(-d_tau * mpo_pool.coeff[chosen_idx]);
                        
            direct_sum.from_direct_sum(walker[i],res_walker);
            direct_sum.compress_svd(walker[i].bond_dim);

            walker[i].careless_copy(direct_sum); 
            */
            double norm_i = walker[i].get_norm();
                // normalize the walker. Why? 
            walker[i].multiply_scalar(1/sqrt(norm_i));  

            walker_coeff[i] *= sqrt(norm_i);
            
            
        }
            
        double temp = get_overlap();

        if (step%20 == 0) 
        {
            cout << "step:" << step <<endl;  
            cout << "overlap after:" << temp <<endl; 

            double energy = get_expecation(); 
            cout <<"ENERGY:" << energy /temp << endl;    

            cout << "walker coeffs:\n";
            for (auto w: walker_coeff)
                cout << w <<endl; 
        }
        // rescale the walker coeff         
        // avoid 
        double coeff_sum = 0; 
        
        for (auto w: walker_coeff)
        {
            coeff_sum +=w; 
        }
        
        for (auto& w: walker_coeff)
            w /= (coeff_sum / num_walker); 

        
    }
}

void sampler::run_compressed()
{
    /*
        run with walker compression
    */   
    vector<pair<int,double>> chosen_op(num_walker); 
    // 1 / |G_k| i.e. group size 
    vector<double> cond_prob(num_walker); 
    

    for (int step = 0; step < num_step;  step++ )
    {
        sample_op(chosen_op, cond_prob); 
        
        for (int i=0; i< num_walker; i++)
        {   
            //  prob of G_k 
            //double p_g = group.range[i+1] - group.range[i]; 
            
            // cond prob of i w. group K chosen 
            //double p_i = 0 ;  
            
            // propagate 
            // use mpo_pool.h_pauli 
            MPS<double> res_walker(walker[i].bond_dim, walker[i].site_num, walker[i].d);
            MPS<double> total,total_2;
            
            
            //cout << "coeff of " << i <<":" << chosen_op[i].second / cond_prob[i] <<endl;

            double spawn_coeff = abs(chosen_op[i].second / cond_prob[i]); 
            double spawn_coeff_floor = floor(spawn_coeff);
            double num_spawn;
            /// span 
            double r = spawn_dice(gen);

            if (r < spawn_coeff - spawn_coeff_floor)
            {
                // spawn event successful.     
                num_spawn = spawn_coeff_floor +1;   
            }
            else 
            {
                num_spawn = spawn_coeff_floor; 
            }

            if (num_spawn!=0)
            {
                walker[i].apply_single_h(mpo_pool.h[chosen_op[i].first],res_walker);

                res_walker.multiply_scalar(-num_spawn);
                // rescale I 
                walker[i].multiply_scalar(num_spawn/(d_tau * mpo_pool.coeff[chosen_op[i].first]));

                total.from_direct_sum(walker[i],res_walker);
                total.compress_svd(walker[i].bond_dim);

                walker[i].careless_copy(total);  

                double norm_i = walker[i].get_norm();
                // normalize the walker. Why? 
                walker[i].multiply_scalar(1/sqrt(norm_i));  

                walker_coeff[i] *= sqrt(norm_i);
            }
                      
            
            //total_2.from_direct_sum(walker[i],res_walker);
            
            
            /*
            cout << i << " compression fidelity:" << total.inner_product(total_2) / total_2.get_norm() <<endl;
            cout << chosen_op[i].second / cond_prob[i] <<endl;
            */
                    
            //walker[i].apply_single_h_inplace(mpo_pool.h_pauli[chosen_op[i].first]);

            // -d_tau * beta / P(G_k) P(i | G_k). normalized 
        
            //walker_coeff[i] *= (chosen_op[i].second / cond_prob[i] );
        } 
        
        //return; 
        double temp = get_overlap();

        if (step%50 == 0)
        {
            cout << "step:" << step <<endl;  
            cout << "overlap after:" << temp <<endl; 

            double energy = get_expecation(); 
            cout <<"ENERGY:" << energy /temp << endl;    
        }
        // rescale the walker coeff         
        double coeff_sum = 0; 
        
        for (auto w: walker_coeff)
        {
            coeff_sum +=w; 
        }
        
        for (auto& w: walker_coeff)
            w /= coeff_sum; 
        
    }
    
}

void sampler::run()
{
    /*  
        execution 
    */
    
    // op_idx, d_tau * alpha 
    vector<pair<int,double>> chosen_op(num_walker); 
    // 1 / |G_k| i.e. group size 
    vector<double> cond_prob(num_walker); 
    

    for (int step = 0; step < num_step;  step++ )
    {
        sample_op(chosen_op, cond_prob); 

        /*
        cout << "beep" <<endl; 
        for (int i=0; i< num_walker; i++)
        {
            cout << chosen_op[i].first << " " << chosen_op[i].second << " " << cond_prob[i] <<endl;   
        } 
        */
        //cout<< "overlap before:" << get_overlap() << endl; 

        for (int i=0; i< num_walker; i++)
        {   
            //  prob of G_k 
            //double p_g = group.range[i+1] - group.range[i]; 
            
            // cond prob of i w. group K chosen 
            //double p_i = 0 ;  
            
            // propagate 
            // use mpo_pool.h_pauli 
            walker[i].apply_single_h_inplace(mpo_pool.h_pauli[chosen_op[i].first]);

            // -d_tau * beta / P(G_k) P(i | G_k). normalized 
        
            walker_coeff[i] *= (chosen_op[i].second / cond_prob[i] );
        } 

        double temp = get_overlap();

        cout << "step:" << step <<endl;  
        cout << "overlap after:" << temp <<endl; 

        double energy = get_expecation(); 
        cout <<"ENERGY:" << energy /temp << endl;    

        // rescale the walker coeff
        for (auto& w: walker_coeff)
        {
            w/=temp; 
        }
        
    }
    

}
void sampler::sample_op(vector<pair<int,double>>& chosen_op,
                        vector<double>& cond_prob) 
{
    /*
        sample a group of 
    */ 
    
    for (int i=0; i< chosen_op.size(); i++)
    {
        double v = real_dice(gen); 
        //cout <<"dice:" << v <<endl; 
        int chosen_group;  
        
        double chosen_p; 
        for (int j=0; j<ite_op.range.size()-1; j++)
        {
            if (v >= ite_op.range[j] &&  v < ite_op.range[j+1]) 
            {
                chosen_group = j;
                // p(G_k)
                chosen_p = ite_op.range[j+1] - ite_op.range[j]; 
                
                break; 
            }
        }
        //cout <<"chosen group idx:" << chosen_group <<endl; 

        // sample operator in the given group w. uniform distribution
        // could use some decaying distributions? 
        
        // creation of this engine is cheap (but seeding is not) 
        uniform_int_distribution<int> dice_int(0,ite_op.idx_op[chosen_group].size()-1);
        // chosen_op ~ [0, group_size-1]!!! 
        int chosen_op_idx  = dice_int(gen);
        
        chosen_op[i] = ite_op.idx_op[chosen_group][chosen_op_idx]; 
        
        //  P(i | G_k) = 1/ |G_k| 
        //  P(G_k) P(i | G_k)
        //  uniformly chosen in the group 
        cond_prob[i] =  chosen_p/ ite_op.idx_op[chosen_group].size() ; 
        
        real_dice.reset(); 
    }
}

double sampler::get_overlap()
{   
    
    double sum = 0;     
    
    //cout << "printing walker coeff\n";    
    
    for (int i=0; i< num_walker; i++)
    {
        double t = wf_trial.inner_product(walker[i]);   
        //cout << i << ":" <<  t << ", " << walker_coeff[i]  << endl; 
        //cout << i << ":" << walker_coeff[i] <<endl; 
        sum +=  walker_coeff[i] * t;    
    }
    
    return sum; 
}

double sampler::get_expecation()
{
    /* 
        use mpo_pool.coeff: less terms. 
    */
    
    // un-normalized energy 
    
    double sum = 0; 

    int num_thread = mpo_pool.num_bucket;  
    //vector<MPS<double>> temp_phi(num_walker);

    //  H |phi_T>  
    #pragma omp parallel for num_threads(num_thread)
    for (int i=0; i< mpo_pool.h.size(); i++ )
    {
        wf_trial.apply_single_h(mpo_pool.h[i],trial_buff[i]);
    }

    #pragma omp parallel for num_threads(num_thread) reduction(+:sum)
    for (int i=0; i< mpo_pool.h.size(); i++ )
    {
        double temp =0; 

        for (int j=0; j< num_walker; j++)
        {
            temp += walker_coeff[j] * mpo_pool.coeff[i] * walker[j].inner_product(trial_buff[i]); 
        }
        
        sum += temp; 
    }
    
    return sum;   
}

double sampler::get_ite_energy()
{
    // one step ite and calc energy 
        
    vector<double> ite_psi_coeff(mpo_pool.h.size());
    
    int cnt = 0; 
    // (1 -dt H) |phi>
    for (auto group: ite_op.idx_op)
        for (auto pair: group)
        {
            wf_trial.apply_single_h(mpo_pool.h[pair.first], trial_buff[cnt]);
            ite_psi_coeff[cnt] = pair.second;
            
            cnt ++; 

        }

    



    
}