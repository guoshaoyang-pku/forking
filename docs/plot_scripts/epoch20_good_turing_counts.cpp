// Exact continuation counts for the fixed-val contexts of epoch20 (§55).
// CPU-only, one pass over nested train prefixes; never crosses chunk boundaries.
// Compile: g++ -O3 -std=c++17 epoch20_good_turing_counts.cpp -o count_gt
// Run: count_gt /path/to/data/tokenized_epoch20 > good_turing_counts.csv
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

constexpr uint64_t V=8192, T=2048, B=72, CHUNK=2049;
constexpr std::array<uint64_t,20> EPB={42,56,84,112,168,224,253,337,421,506,590,674,842,1011,1348,2022,2696,3370,5055,6740};
struct State { uint64_t f=0, singleton=0, val=0; };
struct Hash {
  size_t operator()(uint64_t x) const {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
};
std::string shard(const std::string& root, int sid) {
  std::string n=std::to_string(sid);
  return root+"/shard_"+std::string(5-n.size(),'0')+n+".bin";
}
uint64_t context(const std::array<uint16_t,CHUNK>& a, size_t j) {
  // train.py: prev1=[x0,x0,x1,...], prev2=[x0,x1,x0,x1,...].
  uint64_t prev1=a[j==0 ? 0 : j-1], prev2=a[j<2 ? j : j-2];
  return (prev2*V+prev1)*V+a[j];
}
int main(int argc, char** argv) {
  if(argc!=2) throw std::runtime_error("expected tokenized_epoch20 directory");
  const std::string root=argv[1];
  std::unordered_map<uint64_t,uint32_t,Hash> context_ids;
  std::unordered_map<uint64_t,uint32_t,Hash> val_pairs;
  std::vector<State> states;
  std::array<uint16_t,CHUNK> a;
  context_ids.reserve(600000); val_pairs.reserve(600000);
  std::ifstream val(shard(root,21),std::ios::binary);
  for(uint64_t chunk=0;chunk<4*B;chunk++) {
    if(!val.read(reinterpret_cast<char*>(a.data()),sizeof(a))) throw std::runtime_error("short fixed val");
    for(size_t j=0;j<T;j++) {
      uint64_t c=context(a,j);
      auto inserted=context_ids.emplace(c,states.size());
      if(inserted.second) states.push_back(State{});
      states[inserted.first->second].val++;
      val_pairs[c*V+a[j+1]]++;
    }
  }
  std::unordered_map<uint64_t,uint32_t,Hash> pairs;
  pairs.reserve(8000000);
  uint64_t chunks=0, train_val_context_tokens=0; size_t point=0;
  auto start=std::chrono::steady_clock::now();
  std::cout<<std::setprecision(15);
  std::cout<<"epoch_batches,actual_multiplier,train_tokens,val_tokens,val_distinct_contexts,seen_val_mass,novel_context_mass,Q_gt_seen,Q_gt_all,empirical_unseen_cont_seen,empirical_unseen_pair_all,distinct_pairs_for_val_contexts,train_tokens_in_val_contexts,elapsed_seconds\n";
  for(int sid=0;sid<=20 && point<EPB.size();sid++) {
    std::ifstream input(shard(root,sid),std::ios::binary);
    if(!input) throw std::runtime_error("missing train shard");
    // Match _iter_chunks: carry batches across shards, dropping only partial chunks.
    while(point<EPB.size() && input.read(reinterpret_cast<char*>(a.data()),sizeof(a))) {
      for(size_t j=0;j<T;j++) {
        uint64_t c=context(a,j);
        auto found=context_ids.find(c);
        if(found==context_ids.end()) continue;
        State& s=states[found->second]; s.f++; train_val_context_tokens++;
        auto inserted=pairs.emplace(c*V+a[j+1],0);
        uint32_t& count=inserted.first->second;
        if(count==0) s.singleton++; else if(count==1) s.singleton--;
        count++;
      }
      chunks++;
      if(chunks!=EPB[point]*B) continue;
      double novel=0, gt=0, unseen_seen=0;
      for(const auto& s:states) {
        if(s.f) gt+=double(s.val)*double(s.singleton)/s.f;
        else novel+=s.val;
      }
      for(const auto& item:val_pairs) {
        auto c=context_ids.find(item.first/V);
        if(states[c->second].f && pairs.find(item.first)==pairs.end()) unseen_seen+=item.second;
      }
      double n=4*B*T, seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      std::cout<<EPB[point]<<','<<double(EPB[point])/337<<','<<chunks*T<<','<<uint64_t(n)<<','<<states.size()<<','
               <<1-novel/n<<','<<novel/n<<','<<gt/n<<','<<(gt+novel)/n<<','<<unseen_seen/n<<','<<(unseen_seen+novel)/n<<','
               <<pairs.size()<<','<<train_val_context_tokens<<','<<seconds<<std::endl;
      std::cerr<<"point "<<point+1<<"/20, batches="<<EPB[point]<<", Q_seen="<<gt/n<<", novel="<<novel/n<<", seconds="<<seconds<<std::endl;
      point++;
    }
  }
  if(point!=EPB.size()) throw std::runtime_error("training pool exhausted before 20x");
}
