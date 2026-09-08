#include <bits/stdc++.h>
using namespace std;

int main()
{
  set<int> s;
  s.insert(1);
  s.insert(1);
  s.insert(2);
  s.insert(3);
  s.insert(4);
  s.insert(5);
  s.insert(5);
  for (auto it = s.begin(); it != s.end(); it++)
  {
    cout << *it << " ";
  }
  cout << endl;
  auto it2 = s.find(3);
  auto it3 = s.find(5);
  s.erase(it2, it3);
  cout << "Elements in the set after erasing: ";
  for (auto it = s.begin(); it != s.end(); it++)
  {
    cout << *it << " ";
  }
  cout << endl;
}