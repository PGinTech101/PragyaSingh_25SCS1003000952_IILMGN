#include <bits/stdc++.h>
using namespace std;

int main()
{
  int n;
  cout << "Enter a number: ";
  cin >> n;
  vector<int> v;

  cout << "Enter the values: ";
  for (int i = 0; i < n; i++)
  {
    int data;
    cin >> data;
    v.push_back(data);
  }
  cout << endl;
  cout << "The values are: ";
  for (auto x : v)
  {
    cout << x << " ";
  }
  cout << endl;
  vector<int> odd;
  vector<int> even;
  for (int i = 0; i < n; i++)
  {
    if (v.at(i) % 2 == 0)
    {
      even.push_back(v.at(i));
    }
    else
    {
      odd.push_back(v.at(i));
    }
  }
  cout << "The even numbers are: ";
  for (auto x : even)
  {
    cout << x << " ";
  }
  cout << endl;
  cout << "The odd numbers are: ";
  for (auto x : odd)
  {
    cout << x << " ";
  }
  cout << endl;
}