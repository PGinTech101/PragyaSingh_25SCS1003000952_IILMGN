#include <bits/stdc++.h>
using namespace std;

int main()
{
  multiset<int> ms;

  ms.insert(10);
  ms.insert(12);
  ms.insert(20);
  ms.insert(20);
  ms.insert(30);
  ms.insert(80);
  ms.insert(90);
  ms.insert(90);

  cout << "Duplicates are allowed:" << endl;
  for (auto it = ms.begin(); it != ms.end(); it++)
  {
    cout << *it << " ";
  }
  cout << endl;

  auto it2 = ms.find(12);
  auto it3 = ms.find(80);

  ms.erase(it2, it3);

  cout << "Elements in the multiset after erasing:" << endl;
  for (auto it = ms.begin(); it != ms.end(); it++)
  {
    cout << *it << " ";
  }
  cout << endl;
}