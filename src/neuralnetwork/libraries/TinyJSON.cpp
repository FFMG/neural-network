// Licensed to Florent Guelfucci under one or more agreements.
// Florent Guelfucci licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
#include "TinyJSON.h"

#if TJ_INCLUDE_STDVECTOR == 1
#include <algorithm>
#endif

#include <limits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

static constexpr short TJ_MAX_NUMBER_OF_DIGGITS = 19;
static constexpr short TJ_DEFAULT_STRING_READ_SIZE = 10;
static constexpr unsigned int TJ_DEFAULT_STRING_MAX_READ_SIZE = 4294967295;
static constexpr unsigned int TJ_DEFAULT_STRING_MAX_READ_GROW = TJ_DEFAULT_STRING_MAX_READ_SIZE / 2;

static constexpr TJCHAR TJ_NULL_TERMINATOR = '\0';

static constexpr TJCHAR TJ_UTF8_BOM0 = static_cast<TJCHAR>(0x0EF);
static constexpr TJCHAR TJ_UTF8_BOM1 = static_cast<TJCHAR>(0x0BB);
static constexpr TJCHAR TJ_UTF8_BOM2 = static_cast<TJCHAR>(0x0BF);

static constexpr TJCHAR TJ_ESCAPE_QUOTATION = static_cast<TJCHAR>(0x022);       // % x22 / ; "    quotation mark  U+0022
static constexpr TJCHAR TJ_ESCAPE_REVERSE_SOLIDUS = static_cast<TJCHAR>(0x05C); // % x5C / ; \    reverse solidus U+005C
static constexpr TJCHAR TJ_ESCAPE_SOLIDUS = static_cast<TJCHAR>(0x02F);         // % x2F / ; /    solidus         U+002F
static constexpr TJCHAR TJ_ESCAPE_BACKSPACE = static_cast<TJCHAR>(0x008);       // % x62 / ; b    backspace       U+0008
static constexpr TJCHAR TJ_ESCAPE_FORM_FEED = static_cast<TJCHAR>(0x00C);       // % x66 / ; f    form feed       U+000C
static constexpr TJCHAR TJ_ESCAPE_LINE_FEED = static_cast<TJCHAR>(0x00A);       // % x6E / ; n    line feed       U+000A
static constexpr TJCHAR TJ_ESCAPE_CARRIAGE_RETURN = static_cast<TJCHAR>(0x00D); // % x72 / ; r    carriage return U+000D
static constexpr TJCHAR TJ_ESCAPE_TAB = static_cast<TJCHAR>(0x009);             // % x74 / ; t    tab             U+0009
// static constexpr TJCHAR TJ_ESCAPE_HEXDIG = '\u1234';// % x75 4HEXDIG; uXXXX                U + XXXX

#ifdef _DEBUG
# if defined(_MSC_VER)
#   define TJASSERT(x) for(;;){ if(!x){ __debugbreak();}; break;}
#else
#   include  <assert.h>
#   define TJASSERT(x) for(;;){ assert(x); break;}
# endif
#else
# define TJASSERT(x) for(;;){break;}
#endif

#define TJ_CASE_SIGN          case '-': \
                              case '+': 

#define TJ_CASE_DIGIT         case '0': \
                              case '1': \
                              case '2': \
                              case '3': \
                              case '4': \
                              case '5': \
                              case '6': \
                              case '7': \
                              case '8': \
                              case '9': 

#define TJ_CASE_HEX           case '0': \
                              case '1': \
                              case '2': \
                              case '3': \
                              case '4': \
                              case '5': \
                              case '6': \
                              case '7': \
                              case '8': \
                              case '9': \
                              case 'a': \
                              case 'b': \
                              case 'c': \
                              case 'd': \
                              case 'e': \
                              case 'f': \
                              case 'A': \
                              case 'B': \
                              case 'C': \
                              case 'D': \
                              case 'E': \
                              case 'F': 

#define TJ_CASE_SPACE         case ' ':  \
                              case TJ_ESCAPE_TAB: \
                              case TJ_ESCAPE_LINE_FEED: \
                              case TJ_ESCAPE_CARRIAGE_RETURN:

#define TJ_CASE_START_STRING  case TJ_ESCAPE_QUOTATION:

#define TJ_CASE_COMMA         case ',':

#define TJ_CASE_COLON         case ':':

#define TJ_CASE_BEGIN_OBJECT  case '{':

#define TJ_CASE_END_OBJECT    case '}':

#define TJ_CASE_BEGIN_ARRAY   case '[':

#define TJ_CASE_END_ARRAY     case ']':

#define TJ_CASE_MAYBE_ESCAPE  case '\\':

namespace TinyJSON
{
  struct internal_dump_configuration
  {
    TJCHAR* _buffer;
    const formating _formating;
    const TJCHAR* _indent;
    int _buffer_pos;
    int _buffer_max_length;

    const TJCHAR* _item_separator;
    const TJCHAR* _key_separator;
    const TJCHAR* _value_quote;
    const TJCHAR* _key_quote;
    const TJCHAR* _new_line;
    const bool _escape_special_characters;

    internal_dump_configuration(
      formating formating,
      const TJCHAR* indent,
      const TJCHAR* item_separator,
      const TJCHAR* key_separator,
      const TJCHAR* value_quote,
      const TJCHAR* key_quote,
      const TJCHAR* new_line,
      bool escape_special_characters
    ) :
      _formating(formating),
      _indent(indent),
      _item_separator(item_separator),
      _key_separator(key_separator),
      _value_quote(value_quote),
      _key_quote(key_quote),
      _new_line(new_line),
      _escape_special_characters(escape_special_characters)
    {
      _buffer = nullptr;
      _buffer_max_length = _buffer_pos = 0;
    }
  };

  ///////////////////////////////////////
  // Parse result.
  class ParseResult
  {
  public:
    ParseResult(const parse_options& parse_options) :
      _exception_message(nullptr),
      _options(parse_options),
      _depth(0)
    {
    }

    ParseResult(const ParseResult& parse_result) = delete;
    ParseResult& operator=(const ParseResult& parse_result) = delete;

    ~ParseResult()
    {
      free_exception_message();
    }

    void push_depth()
    {
      ++_depth;
    }

    void pop_depth()
    {
      --_depth;
    }

    unsigned int current_depth() const
    {
      return _depth;
    }

    /// <summary>
    /// Assign a parse error message.
    /// </summary>
    /// <param name="parse_exception_message"></param>
    void assign_exception_message(const char* parse_exception_message)
    {
      free_exception_message();
      if (parse_exception_message != nullptr)
      {
        auto length = strlen(parse_exception_message);
        _exception_message = new char[length + 1];
        std::strcpy(_exception_message, parse_exception_message);
      }
    }

    void throw_if_exception()
    {
      if (!_options.throw_exception)
      {
        return;
      }
      if (!has_exception_message())
      {
        return;
      }
      throw TJParseException(_exception_message);
    }

    bool has_exception_message() const
    {
      return _exception_message != nullptr;
    }

    const parse_options& options() const
    {
      return _options;
    }

  protected:
    void free_exception_message() noexcept
    {
      if (_exception_message != nullptr)
      {
        delete[] _exception_message;
        _exception_message = nullptr;
      }
    }

    char* _exception_message;
    const parse_options& _options;
    unsigned int _depth;
  };

  ///////////////////////////////////////
  // Write result.
  class WriteResult
  {
  public:
    WriteResult(const write_options& write_options) :
      _exception_message(nullptr),
      _options(write_options)
    {
    }

    WriteResult(const WriteResult& parse_result) = delete;
    WriteResult& operator=(const WriteResult& parse_result) = delete;

    ~WriteResult()
    {
      free_exception_message();
    }

    /// <summary>
    /// Assign a write error message.
    /// </summary>
    /// <param name="write_exception_message"></param>
    void assign_exception_message(const char* write_exception_message)
    {
      free_exception_message();
      if (write_exception_message != nullptr)
      {
        auto length = strlen(write_exception_message);
        _exception_message = new char[length + 1];
        std::strcpy(_exception_message, write_exception_message);
      }
    }

    void throw_if_exception()
    {
      if (!_options.throw_exception)
      {
        return;
      }
      if (!has_exception_message())
      {
        return;
      }
      throw TJWriteException(_exception_message);
    }

    bool has_exception_message() const
    {
      return _exception_message != nullptr;
    }

    const write_options& options() const
    {
      return _options;
    }

  protected:
    void free_exception_message() noexcept
    {
      if (_exception_message != nullptr)
      {
        delete[] _exception_message;
        _exception_message = nullptr;
      }
    }

    char* _exception_message;
    const write_options& _options;
  };

  ///////////////////////////////////////
  /// Parsing Exception
  TJParseException::TJParseException(const char* message) :
    _message(nullptr)
  {
    assign_message(message);
  }

  TJParseException::TJParseException(const TJParseException& exception)
    : _message(nullptr)
  {
    *this = exception;
  }

  TJParseException::~TJParseException()
  {
    free_message();
  }

  TJParseException& TJParseException::operator=(const TJParseException& exception)
  {
    if (this != &exception)
    {
      assign_message(exception._message);
    }
    return *this;
  }

  const char* TJParseException::what() const noexcept
  {
    return _message == nullptr ? "Unknown" : _message;
  }

  void TJParseException::assign_message(const char* message)
  {
    free_message();
    if (message != nullptr)
    {
      auto length = strlen(message);
      _message = new char[length + 1];
      std::strcpy(_message, message);
    }
  }

  void TJParseException::free_message() noexcept
  {
    if (_message != nullptr)
    {
      delete[] _message;
      _message = nullptr;
    }
  }

  ///////////////////////////////////////
  /// Write Exception
  TJWriteException::TJWriteException(const char* message) :
    _message(nullptr)
  {
    assign_message(message);
  }

  TJWriteException::TJWriteException(const TJWriteException& exception)
    : _message(nullptr)
  {
    *this = exception;
  }

  TJWriteException::~TJWriteException()
  {
    free_message();
  }

  TJWriteException& TJWriteException::operator=(const TJWriteException& exception)
  {
    if (this != &exception)
    {
      assign_message(exception._message);
    }
    return *this;
  }

  const char* TJWriteException::what() const noexcept
  {
    return _message == nullptr ? "Unknown" : _message;
  }

  void TJWriteException::assign_message(const char* message)
  {
    free_message();
    if (message != nullptr)
    {
      auto length = strlen(message);
      _message = new char[length + 1];
      std::strcpy(_message, message);
    }
  }

  void TJWriteException::free_message() noexcept
  {
    if (_message != nullptr)
    {
      delete[] _message;
      _message = nullptr;
    }
  }

#if TJ_INCLUDE_STDVECTOR != 1
  ///////////////////////////////////////
  /// Protected Array class.
  class TJList
  {
  public:
    TJList()
      :
      _values(nullptr),
      _number_of_items(0),
      _capacity(1)
    {
    }

    virtual ~TJList()
    {
      clean();
    }

    /// <summary>
    /// Add an item to our array, grow if needed.
    /// </summary>
    /// <param name="value"></param>
    void add(TJValue* value)
    {
      if (nullptr == _values)
      {
        //  first time using the array.
        _values = new TJValue*[_capacity];
      }
      if (_number_of_items == _capacity)
      {
        grow();
      }
      _values[_number_of_items++] = value;
    }

    /// <summary>
    /// Get the size of the array, (not the capacity).
    /// </summary>
    /// <returns></returns>
    unsigned int size() const
    {
      return _number_of_items;
    }

    /// <summary>
    /// Get a value at an index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    TJValue* at(unsigned int index) const
    {
      return index < _number_of_items ? _values[index] : nullptr;
    }
  private:
    /// <summary>
    /// The pointers we will take ownership of.
    /// </summary>
    TJValue** _values;

    /// <summary>
    /// The number of items in the array
    /// </summary>
    unsigned int _number_of_items;

    /// <summary>
    /// The capacity
    /// </summary>
    unsigned int _capacity;

    /// <summary>
    /// Delete all the pointers in the array and the array itself.
    /// </summary>
    void clean()
    {
      if (nullptr == _values)
      {
        return;
      }
      for (unsigned int i = 0; i < _number_of_items; ++i)
      {
        delete _values[i];
      }
      delete[] _values;
      _values = nullptr;
    }

    /// <summary>
    /// Grow the array by nultiplying the capacity by 2.
    /// </summary>
    void grow()
    {
      _capacity = _capacity << 1;

      // create the new container
      TJValue** temp_values = new TJValue*[_capacity];

      // just move the data from one to the other as we wil take ownership of it.
      memmove(temp_values, _values, _number_of_items * sizeof(TJValue*));

      // clean up the old value and point it to the temp value.
      delete[] _values;
      _values = temp_values;
    }

    // no copies.
    TJList(const TJList&) = delete;
    TJList& operator=(const TJList&) = delete;
    TJList& operator=(TJList&&) = delete;
  };

  ///////////////////////////////////////
  /// Protected Array class.
  class TJDictionary
  {
  private:
    /// <summary>
    /// The dictionary data, the key and the index of the value
    /// </summary>
    struct dictionary_data
    {
      unsigned int _value_index;
      const TJCHAR* _key;
    };

    /// <summary>
    /// Structure that return the dictionary index, (or the nearest one)
    /// and the flag 'was_found' tells us if the actual value was located.
    /// </summary>
    struct search_result {
      int _dictionary_index;
      bool _was_found;
    };

  public:
    TJDictionary()
      :
      _values(nullptr),
      _values_dictionary_cs(nullptr),
      _values_dictionary_ci(nullptr),
      _number_of_items(0),
      _number_of_items_dictionary_cs(0),
      _number_of_items_dictionary_ci(0),
      _capacity(1)
    {
    }

    virtual ~TJDictionary()
    {
      clean();
    }

    /// <summary>
    /// Remove a value at a certain value.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    bool pop(const TJCHAR* key)
    {
      // get the index of both values, the case sentitive one
      // is the one we want to remove
      auto binary_search_result_cs = binary_search(key, true);
      auto dictionary_index_cs = binary_search_result_cs._dictionary_index;
      int value_index_cs = binary_search_result_cs._was_found ? _values_dictionary_cs[dictionary_index_cs]._value_index : -1;

      // the case insensitive one is a little more complex.
      // if we have "a" and "A" in our database and we want to pop("a")
      // then we have to make sure that we get the correct one.
      auto binary_search_result_ci = binary_search(key, false);
      auto dictionary_index_ci = binary_search_result_ci._dictionary_index;
      int value_index_ci = binary_search_result_ci._was_found ? _values_dictionary_cs[dictionary_index_ci]._value_index : -1;

      // if we have no indexes then we have noting to pop.
      if (value_index_cs == -1)
      {
        TJASSERT(value_index_ci == -1); // how can it be???
                                        // surely if we do not have one we do not have the other 
        return false;
      }

      // if both indexes are the same, then we can just remove them.
      if (value_index_cs == value_index_ci)
      {
        /// life is good, we are all pointing at the same code.
        // so we can remove both values at the indexes we found aleady.
        remove_dictionary_data_by_dictionary_index
        (
          binary_search_result_cs._dictionary_index,
          _number_of_items_dictionary_cs,
          _values_dictionary_cs
        );
        remove_dictionary_data_by_dictionary_index
        (
          binary_search_result_ci._dictionary_index,
          _number_of_items_dictionary_ci,
          _values_dictionary_ci
        );
      }
      else
      if (value_index_cs != value_index_ci)
      {
        // this is a bit more difficult, while we found the exact match
        // there seem to be a case insensitive match as well.
        // so we now have to remove it by index.

        // we know that the case sensitive one was found ... so it can be removed.
        remove_dictionary_data_by_dictionary_index
        (
          binary_search_result_cs._dictionary_index,
          _number_of_items_dictionary_cs,
          _values_dictionary_cs
        );

        // the issue is the one that is not case sensitive, we nee to remove the correct one.
        remove_dictionary_data_by_value_index(
          value_index_cs,
          _number_of_items_dictionary_ci,
          _values_dictionary_ci
        );
      }

      // shift the value to the right and update the counter.
      shift_value_right(value_index_cs);
      --_number_of_items;
      return true;
    }

    /// <summary>
    /// Initialise all the values.
    /// We only do it when needed.
    /// </summary>
    void initilize()
    {
      _values = new TJMember * [_capacity];
      _values_dictionary_cs = new dictionary_data[_capacity];
      _values_dictionary_ci = new dictionary_data[_capacity];
      _number_of_items = 0;
      _number_of_items_dictionary_cs = 0;
      _number_of_items_dictionary_ci = 0;
    }

    /// <summary>
    /// set a value in our dictionary, if the value exists, (by name)
    /// we will replace the old value with the new value.
    /// </summary>
    void set(TJMember* value)
    {
      const TJCHAR* key = value->name();
      if (nullptr == _values)
      {
        initilize();
      }

      // check if the key already exists, if it does simply update the value.
      // we need to update the values in both case sensitive and insensitive
      auto binary_search_result_cs = binary_search(key, true);
      if (true == binary_search_result_cs._was_found)
      {
        replace_dictionaries_data(value, binary_search_result_cs._dictionary_index);
        return;
      }

      // check if the value can be added.
      // if not grow both the dictionary and the value.
      if (_number_of_items == _capacity)
      {
        grow();
      }

      // get the value index.
      auto value_index = _number_of_items++;
      _values[value_index] = value;

      ///
      // Add the item to the case sensitive dictionary
      // 
      //  shift everything to the left.
      auto dictionary_index_cs = binary_search_result_cs._dictionary_index;
      // check that we will have space.
      TJASSERT((_number_of_items_dictionary_cs + 1) >= _capacity);

      // add the dictionary index value
      add_dictionary_data(
        key, 
        value_index, 
        dictionary_index_cs,
        _values_dictionary_cs,
        _number_of_items_dictionary_cs
      );

      // at this point both values should be the same for the case sensitive.
      // this is because exact matches always match
      TJASSERT(_number_of_items == _number_of_items_dictionary_cs);

      // we will also need the ci data
      auto binary_search_result_ci = binary_search(key, false);

      //  shift everything to the left.
      auto dictionary_index_ci = binary_search_result_ci._dictionary_index;

      // if this item exists already, we are insertng at the same spot
      // but it does not matter as we will be shifting things around.

      // check that we will have space.
      TJASSERT((_number_of_items_dictionary_ci + 1) >= _capacity);

      // add the dictionary index value
      add_dictionary_data(
        key,
        value_index,
        dictionary_index_ci,
        _values_dictionary_ci,
        _number_of_items_dictionary_ci
      );
    }

    /// <summary>
    /// Get the size of the array, (not the capacity).
    /// </summary>
    /// <returns></returns>
    unsigned int size() const
    {
      return _number_of_items;
    }

    /// <summary>
    /// Get a value at an index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    TJMember* at(unsigned int index) const
    {
      return index < _number_of_items ? _values[index] : nullptr;
    }

    /// <summary>
    /// Find a single value using the given lambda function to compare,
    /// </summary>
    /// <typeparam name="Compare"></typeparam>
    /// <param name="compare"></param>
    /// <returns></returns>
    TJMember* at(const TJCHAR* key, bool case_sensitive) const
    {
      // look in the dictionary depending on the case sensitivity.
      auto binary_search_result = binary_search(key, case_sensitive);

      // if we found it, return the actual index value.
      int index = binary_search_result._was_found ? 
        ( case_sensitive ? 
          _values_dictionary_cs[binary_search_result._dictionary_index]._value_index 
          :
          _values_dictionary_ci[binary_search_result._dictionary_index]._value_index
        )
        : -1;

      return index != -1 ? _values[index] : nullptr;
    }

  private:
    /// <summary>
    /// The pointers we will take ownership of.
    /// </summary>
    TJMember** _values;

    /// <summary>
    /// The key value pairs to help case sensitive binary search.
    /// </summary>
    dictionary_data* _values_dictionary_cs;

    /// <summary>
    /// The key value pairs to help case insensitive binary search.
    /// </summary>
    dictionary_data* _values_dictionary_ci;

    /// <summary>
    /// The number of items in the array
    /// </summary>
    unsigned int _number_of_items;

    /// <summary>
    /// The current number of items in the case sensitive dictionary
    /// This might not always be the same, (for a brief time)
    /// As the number of items
    /// </summary>
    unsigned int _number_of_items_dictionary_cs;

    /// <summary>
    /// The current number of items in the case insensitive dictionary
    /// This might not always be the same, (for a brief time)
    /// As the number of items
    /// </summary>
    unsigned int _number_of_items_dictionary_ci;

    /// <summary>
    /// The capacity of both the dictionary and the values.
    /// When we grow the value we grow both dictionaries as well.
    /// </summary>
    unsigned int _capacity;

    /// <summary>
    /// Update the value of the key for a given dictionary.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="dictionary_index"></param>
    /// <param name="dictionary"></param>
    static void replace_dictionary_data
    (
      const TJCHAR* key,
      unsigned int dictionary_index,
      dictionary_data*& dictionary
    )
    {
      // we also need to update the name as we are now pointing somewhere else
      dictionary[dictionary_index]._key = key;
    }

    /// <summary>
    /// Delete all the pointers in the arrays and the array themselves.
    /// </summary>
    void clean()
    {
      if (nullptr == _values)
      {
        return;
      }

      TJASSERT(_number_of_items == _number_of_items_dictionary_cs);
      for (unsigned int i = 0; i < _number_of_items; ++i)
      {
        delete _values[i];
      }

      // clean up the old values and point it to the temp values.
      delete[] _values;
      _values = nullptr;

      delete[] _values_dictionary_cs;
      _values_dictionary_cs = nullptr;

      delete[] _values_dictionary_ci;
      _values_dictionary_ci = nullptr;
    }

    /// <summary>
    /// Grow the array and the dictionary array by nultiplying the capacity by 2.
    /// </summary>
    void grow()
    {
      // grow the capacity
      _capacity = _capacity << 1;
      TJASSERT(_capacity > _number_of_items);

      // create the new containers as temp containers
      auto temp_values = new TJMember*[_capacity];
      auto temp_values_dictionary_cs = new dictionary_data[_capacity];
      auto temp_values_dictionary_ci = new dictionary_data[_capacity];

      // just move the data from one to the other as we will take ownership of it.
      memmove(temp_values, _values, _number_of_items * sizeof(TJMember*));
      memmove(temp_values_dictionary_cs, _values_dictionary_cs, _number_of_items_dictionary_cs * sizeof(dictionary_data));
      memmove(temp_values_dictionary_ci, _values_dictionary_ci, _number_of_items_dictionary_ci * sizeof(dictionary_data));

      // replace the old values
      delete[] _values;
      _values = temp_values;

      // replace the old dictionary values.
      delete[] _values_dictionary_cs;
      _values_dictionary_cs = temp_values_dictionary_cs;

      // replace the old dictionary values.
      delete[] _values_dictionary_ci;
      _values_dictionary_ci = temp_values_dictionary_ci;
    }

    /// <summary>
    /// Remove a directory index at a given entry
    /// </summary>
    /// <param name="key">The key we are looking to remove.</param>
    /// <param name="case_sensitive">If we want to remove it from the case sensitive dictionary or not.</param>
    bool remove_dictionary_data(const TJCHAR* key, bool case_sensitive)
    {
      // search for the key 
      auto binary_search_result = binary_search(key, case_sensitive);
      if (false == binary_search_result._was_found)
      {
        return false;
      }

      if (case_sensitive == true)
      {
        remove_dictionary_data_by_dictionary_index
        (
          binary_search_result._dictionary_index,
          _number_of_items_dictionary_cs,
          _values_dictionary_cs
        );
        return true;
      }

      remove_dictionary_data_by_dictionary_index
      (
        binary_search_result._dictionary_index,
        _number_of_items_dictionary_ci,
        _values_dictionary_ci
      );
      return true;
    }

    /// <summary>
    /// Remove a dictionary entry at a given index.
    /// </summary>
    /// <param name="dictionary_index">The dictionary index.</param>
    /// <param name="dictionary_size"></param>
    /// <param name="dictionary">The dictionary we will be looking in</param>
    static void remove_dictionary_data_by_dictionary_index
    (
      const unsigned int& dictionary_index,
      unsigned int& dictionary_size,
      dictionary_data*& dictionary
    )
    {
      auto index = dictionary[dictionary_index]._value_index;
      remove_dictionary_data(
        dictionary_index,
        dictionary,
        dictionary_size
      );

      // finally we need to move all the index _after_ the dictionary index down by one.
      uppdate_dictionary_data_by_value_index(
        index,
        dictionary_size,
        dictionary
      );
    }

    /// <summary>
    /// Remove a dictionary entry given the value index we are looking for.
    /// </summary>
    /// <param name="index">The value index we are looking for.</param>
    /// <param name="dictionary_size"></param>
    /// <param name="dictionary">The dictionary we will be looking in</param>
    static void remove_dictionary_data_by_value_index(
      const unsigned int& index,
      unsigned int& dictionary_size,
      dictionary_data*& dictionary
    )
    {
      // finally we need to move all the index _after_ the dictionary index down by one.
      for (unsigned int i = 0; i < dictionary_size; ++i)
      {
        auto value_index = dictionary[i]._value_index;
        if (value_index != index)
        {
          continue;
        }
        shift_dictionary_right(value_index, dictionary, dictionary_size);

        // update the counter
        --dictionary_size;

        // finally we need to move all the index _after_ the dictionary index down by one.
        uppdate_dictionary_data_by_value_index(
          value_index,
          dictionary_size,
          dictionary
        );

        // we have to get out as we removed them one and only.
        return;
      }
    }

    /// <summary>
    /// As we removed a value index we need to shift all the value indexes
    /// That are past the value index.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="dictionary_size"></param>
    /// <param name="dictionary"></param>
    static void uppdate_dictionary_data_by_value_index(
      const unsigned int& index,
      const unsigned int& dictionary_size,
      dictionary_data*& dictionary
    )
    {
      // finally we need to move all the index _after_ the dictionary index down by one.
      for (unsigned int i = 0; i < dictionary_size; ++i)
      {
        if (dictionary[i]._value_index >= index)
        {
          --dictionary[i]._value_index;
        }
      }
    }
    
    /// <summary>
    /// Replace both the dictionarry data.
    /// </summary>
    /// <param name="value"></param>
    /// <param name="dictionary_index_cs"></param>
    void replace_dictionaries_data(TJMember* value, unsigned int dictionary_index_cs)
    {
      // we need to search the case insensitive one now
      // this is because the key is actually shared around.
      // so we first look for it ... then we replace it.
      auto binary_search_result_ci = binary_search(value->name(), false);

      auto index = _values_dictionary_cs[dictionary_index_cs]._value_index;
      if (_values[index] != value)
      {
        delete _values[index];
        _values[index] = value;
      }
      replace_dictionary_data(value->name(), dictionary_index_cs, _values_dictionary_cs);

      // remove the case insensitive
      auto dictionary_index_ci = binary_search_result_ci._dictionary_index;
      TJASSERT(true == binary_search_result_ci._was_found);
      if (true == binary_search_result_ci._was_found)
      {
        TJASSERT(dictionary_index_ci == dictionary_index_cs);
        replace_dictionary_data(value->name(), dictionary_index_ci, _values_dictionary_ci);
      }
    }

    /// <summary>
    /// Remove a single dictionary entry shift the data.
    /// </summary>
    /// <param name="dictionary_index"></param>
    /// <param name="dictionary"></param>
    /// <param name="dictionary_size"></param>
    static void remove_dictionary_data(
      unsigned int dictionary_index,
      dictionary_data*& dictionary,
      unsigned int& dictionary_size
    )
    {
      shift_dictionary_right(dictionary_index, dictionary, dictionary_size);

      // we don't have any cleanup to do so we just subtract the index.
      --dictionary_size;
    }

    /// <summary>
    /// Add the value to the dictionary
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value_index"></param>
    /// <param name="dictionary_index"></param>
    static void add_dictionary_data(
      const TJCHAR* key, 
      unsigned int value_index, 
      unsigned int dictionary_index,
      dictionary_data*& dictionary,
      unsigned int& dictionary_size
    )
    {
      shift_dictionary_left(
        dictionary_index,
        dictionary,
        dictionary_size
      );

      // build the new dictionary dta data
      dictionary_data dictionary_data = { value_index, key };

      // finally set the dictionary at the correct value
      dictionary[dictionary_index] = dictionary_data;
      ++dictionary_size;
    }

    /// <summary>
    /// Custom case compare that probably will not work with
    /// locals and so on.
    /// </summary>
    /// <param name="s1"></param>
    /// <param name="s2"></param>
    /// <returns></returns>
    static int case_compare(const TJCHAR* s1, const TJCHAR* s2, bool case_sensitive)
    {
      if (true == case_sensitive)
      {
        return strcmp(s1, s2);
      }

      while (tolower(*s1) == tolower(*s2))
      {
        if (*s1++ == TJ_NULL_TERMINATOR)
        {
          return 0;
        }
        ++s2;
      }
      return tolower(*s1) - tolower(*s2);
    }

    /// <summary>
    /// Do a binary search and return either the exact location of the item
    /// or the location of the item we should insert in the dictionary if we want to keep 
    /// the key order valid.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="case_sensitive">If the search is case sensitive or not.</param>
    /// <returns></returns>
    search_result binary_search(const TJCHAR* key, bool case_sensitive) const
    {
      if (case_sensitive == true)
      {
        return binary_search(key, _values_dictionary_cs, _number_of_items_dictionary_cs, true);
      }
      return binary_search(key, _values_dictionary_ci, _number_of_items_dictionary_ci, false);
    }

    /// <summary>
    /// Do a binary search and return either the exact location of the item
    /// or the location of the item we should insert in the dictionary if we want to keep 
    /// the key order valid.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="dictionary"></param>
    /// <param name="dictionary_size"></param>
    /// <param name="case_sensitive"></param>
    /// <returns></returns>
    static search_result binary_search(
      const TJCHAR* key, 
      const dictionary_data* dictionary,
      const unsigned int& dictionary_size,
      bool case_sensitive
    )
    {
      if (dictionary_size == 0)
      {
        //  we have no data, so we have to put it in the first place.
        search_result result = {};
        result._was_found = false;
        result._dictionary_index = 0;
        return result;
      }

      int first = 0;
      int last = dictionary_size - 1;
      int middle = 0;
      while (first <= last)
      {
        // the middle is the floor.
        middle = static_cast<unsigned int>(first + (last - first) / 2);
        // we do not want duplicate keys
        auto compare = case_compare(dictionary[middle]._key, key, case_sensitive);
        if (compare == 0)
        {
          search_result result = {};
          result._was_found = true;
          result._dictionary_index = middle;
          return result;
        }
        if (compare < 0)
        {
          first = middle + 1;
        }
        else
        {
          last = middle - 1;
        }
      }

      search_result result = {};
      result._was_found = false;
      result._dictionary_index = first;
      return result;
    }

    /// <summary>
    /// Shift everything one position to the right from the index value given.
    /// </summary>
    /// <param name="dictionary_index"></param>
    static void shift_dictionary_right(
      int dictionary_index,
      dictionary_data*& dictionary,
      const unsigned int& dictionary_size
    )
    {
      if (dictionary_size == 0)
      {
        return;
      }

      // shift everything in memory a little to the left.
      memmove(
        &dictionary[dictionary_index],                                  // we are moving +1 to the left
        &dictionary[dictionary_index+1],                                      // we are moving from here.
        (dictionary_size - dictionary_index -1) * sizeof(dictionary_data)); // we are moving the total number of elements less were we are shifting from.
    }

    /// <summary>
    /// Shift everything one position to the right from the index value given.
    /// </summary>
    /// <param name="value_index"></param>
    void shift_value_right(int value_index)
    {
      if (_number_of_items == 0)
      {
        return;
      }

      delete _values[value_index];

      // shift everything in memory a little to the left.
      memmove(
        &_values[value_index],                                  // we are moving +1 to the left
        &_values[value_index + 1],                                      // we are moving from here.
        (_number_of_items - value_index - 1) * sizeof(TJMember*)); // we are moving the total number of elements less were we are shifting from.
    }

    /// <summary>
    /// Shift everything one position to the left from the index value given.
    /// </summary>
    /// <param name="dictionary_index"></param>
    static void shift_dictionary_left(
      int dictionary_index,
      dictionary_data*& dictionary,
      const unsigned int& dictionary_size
      )
    {
      if (dictionary_size == 0)
      {
        return;
      }

      // shift everything in memory a little to the left.
      memmove(
        &dictionary[dictionary_index + 1],                                  // we are moving +1 to the left
        &dictionary[dictionary_index],                                      // we are moving from here.
        (dictionary_size - dictionary_index) * sizeof(dictionary_data)); // we are moving the total number of elements less were we are shifting from.
    }

    // no copies.
    TJDictionary(const TJDictionary&) = delete;
    TJDictionary& operator=(const TJDictionary&) = delete;
    TJDictionary& operator=(TJDictionary&&) = delete;
  };
#endif

  ///////////////////////////////////////
  /// Protected Helper class
  class TJHelper
  {
    friend TJ;
    friend TJMember;
    friend TJValue;
    friend TJValueArray;
    friend TJValueBoolean;
    friend TJValueNumberExponent;
    friend TJValueNull;
    friend TJValueNumberFloat;
    friend TJValueNumberInt;
    friend TJValueObject;
    friend TJValueString;
  protected:
    // Function to multiply an unsigned integer by 10 using bit-shifting
    static unsigned long long fast_multiply_by_10(unsigned long long number) 
    {
      return (number << 3) + (number << 1);
    }

    static unsigned long long fast_power_of_10(unsigned int exponent)
    {
      if (exponent == 0)
      {
        return 1;
      }

      unsigned long long base = 10;
      for (unsigned int i = 1; i < exponent; ++i)
      {
        base = fast_multiply_by_10(base);
      }
      return base;
    }

    /// <summary>
    /// Raise a number to the power of 16
    /// </summary>
    /// <param name="number"></param>
    /// <param name="exponent"></param>
    /// <returns></returns>
    static unsigned long long fast_power_of_16(unsigned int exponent)
    {
      if (exponent == 0)
      {
        return 1;
      }

      unsigned long long base = 16;
      for (unsigned int i = 1; i < exponent; ++i)
      {
        base = base << 4;
      }
      return base;
    }

    /// <summary>
    /// Get the length of a string.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    static unsigned int string_length(const TJCHAR* source)
    {
      if (nullptr == source)
      {
        return 0;
      }

      for (auto i = 0;; ++i)
      {
        if (source[i] == TJ_NULL_TERMINATOR)
        {
          return i;
        }
      }
      return 0;
    }

    /// <summary>
    /// Quickly convert an +ve integer to a string then add -ve if we want.
    /// we also add leading zeros if needed.
    /// </summary>
    /// <param name="number"></param>
    /// <param name="fraction_exponent"></param>
    /// <param name="is_negative"></param>
    /// <param name="length"></param>
    /// <returns></returns>
    static TJCHAR* fast_number_to_string(unsigned long long number, unsigned int fraction_exponent, bool is_negative, unsigned int& length, bool force_positive_sign = false)
    {
      TJCHAR reverse_buffer[255];
      unsigned reverse_position = 0;
      length = 0;

      if (number == 0)
      {
        reverse_buffer[reverse_position++] = '0';
      }
      else
      {
        while (number > 0)
        {
          auto mod = number % 10;
          switch (mod)
          {
          case 0:
            reverse_buffer[reverse_position++] = '0';
            break;

          case 1:
            reverse_buffer[reverse_position++] = '1';
            break;

          case 2:
            reverse_buffer[reverse_position++] = '2';
            break;
          case 3:
            reverse_buffer[reverse_position++] = '3';
            break;
          case 4:
            reverse_buffer[reverse_position++] = '4';
            break;
          case 5:
            reverse_buffer[reverse_position++] = '5';
            break;
          case 6:
            reverse_buffer[reverse_position++] = '6';
            break;
          case 7:
            reverse_buffer[reverse_position++] = '7';
            break;
          case 8:
            reverse_buffer[reverse_position++] = '8';
            break;
          case 9:
            reverse_buffer[reverse_position++] = '9';
            break;

          default:
            break;
          }
          number /= 10;
        }
      }

      if (fraction_exponent > reverse_position)
      {
        const unsigned int& zeros = fraction_exponent - reverse_position;
        for (unsigned int j = 0; j < zeros; ++j)
        {
          reverse_buffer[reverse_position++] = '0';
        }
      }
      if (is_negative)
      {
        reverse_buffer[reverse_position++] = '-';
      }
      else if (force_positive_sign)
      {
        reverse_buffer[reverse_position++] = '+';
      }

      TJCHAR* buffer = new TJCHAR[reverse_position+1];
      buffer[reverse_position] = TJ_NULL_TERMINATOR;
      for (unsigned int i = 0; i < reverse_position; ++i)
      {
        buffer[reverse_position -1 - i] = reverse_buffer[i];
      }

      length = reverse_position;
      return buffer;
    }

    static TJCHAR* fast_number_to_string(unsigned long long number, unsigned int fraction_exponent, bool is_negative, bool force_positive_sign = false)
    {
      unsigned int ignore = 0;
      return fast_number_to_string(number, fraction_exponent, is_negative, ignore, force_positive_sign);
    }

    /// <summary>
    /// 'join' a whole number together with a fraction
    /// If the number is -12.0045 then
    ///   - the number is 12
    ///   - the fraction is 45
    ///   - the exponent is 4
    ///   - and it is negative.
    /// </summary>
    /// <param name="number">The number we are creating</param>
    /// <param name="fraction">The whole number part of the fraction</param>
    /// <param name="fraction_exponent">The length of the fraction, needed for leading zeros.</param>
    /// <param name="is_negative">If the number is negative or not.</param>
    /// <returns></returns>
    static TJCHAR* fast_number_and_fraction_to_string(unsigned long long number, unsigned long long fraction, unsigned int fraction_exponent, bool is_negative)
    {
      // format the number and fraction separately
      unsigned int length_of_number, length_of_fraction;

      // the number has negative sign in front.
      auto string_number = fast_number_to_string(number, 0, is_negative, length_of_number);

      // the fraction does not have a negative sign
      auto string_fraction = fast_number_to_string(fraction, fraction_exponent, false, length_of_fraction);

      // calculate the total length, we add +1 for the '.' and +1 for the null terminator
      int total_length  = length_of_number + length_of_fraction + 1 + 1;
      int final_string_pos = 0;

      // recreate the final string
      TJCHAR* final_string = new TJCHAR[total_length];
      add_string_to_string(string_number, final_string, final_string_pos, total_length);
      add_char_to_string('.', final_string, final_string_pos, total_length);
      add_string_to_string(string_fraction, final_string, final_string_pos, total_length);

      // cleanup the number and fraction.
      delete[] string_number;
      delete[] string_fraction;
      return final_string;
    }

    /// <summary>
    /// 'join' a whole number together with a fraction and add the exponent as well if needed.
    /// If the number is -12.0045e+12 then
    ///   - the number is 12
    ///   - the fraction is 45
    ///   - the fraction_exponent is 4
    ///   - the exponent is 12
    ///   - and the number is negative.
    /// </summary>
    /// <param name="number"></param>
    /// <param name="fraction"></param>
    /// <param name="fraction_exponent"></param>
    /// <param name="exponent"></param>
    /// <param name="is_negative"></param>
    /// <returns></returns>
    static TJCHAR* fast_number_fraction_and_exponent_to_string(unsigned long long number, unsigned long long fraction, unsigned int fraction_exponent, int exponent, bool is_negative)
    {
      // format the number and fraction separately
      unsigned int length_of_number = 0, length_of_fraction = 0, length_of_exponent = 0;

      // the number has negative sign in front.
      auto string_number = fast_number_to_string(number, 0, is_negative, length_of_number);

      // the fraction does not have a negative sign
      TJCHAR* string_fraction = nullptr;
      if (fraction > 0)
      {
        string_fraction = fast_number_to_string(fraction, fraction_exponent, false, length_of_fraction);
      }

      // the fraction does not have a negative sign
      TJCHAR* string_exponent = nullptr;
      if (exponent < 0)
      {
        string_exponent = fast_number_to_string(-1*exponent, 0, true, length_of_exponent);
      }
      else
      {
        string_exponent = fast_number_to_string(exponent, 0, false, length_of_exponent, true);
      }

      // calculate the total length, 
      //   - +1 for the '.' (if needed)
      //   - +1 for the null terminator
      //   - +1 for the 'e' (if needed)
      int total_length = length_of_number + length_of_fraction + length_of_exponent +1 + 1 + 1;
      int final_string_pos = 0;

      // recreate the final string
      TJCHAR* final_string = new TJCHAR[total_length];
      add_string_to_string(string_number, final_string, final_string_pos, total_length);
      if (nullptr != string_fraction)
      {
        add_char_to_string('.', final_string, final_string_pos, total_length);
        add_string_to_string(string_fraction, final_string, final_string_pos, total_length);
      }
      if (nullptr != string_exponent)
      {
        add_char_to_string('e', final_string, final_string_pos, total_length);
        add_string_to_string(string_exponent, final_string, final_string_pos, total_length);
      }

      // cleanup the number and fraction.
      delete[] string_number;
      delete[] string_fraction;
      delete[] string_exponent;
      return final_string;
    }

    /// <summary>
    /// Try and convert a hex string to a number.
    /// If the number returned is -1 then we can assume an error.
    /// This method does not allow, (or expect), 0x at the begining
    /// and does not case about a leading 0
    /// If you pas 123 then we will assume that it is hex and return 291
    /// If you pass 0123 then we will still return 291
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    static long long fast_hex_to_decimal(const TJCHAR* source)
    {
      long long decimal = 0ll;
      auto length = string_length(source);
      auto power = 0;
      for(int i = length-1; i >= 0; --i)
      {
        unsigned int number = 0;
        switch (source[i])
        {
        case '0':
          number = 0;
          break;

        case '1':
          number = 1;
          break;

        case '2':
          number = 2;
          break;

        case '3':
          number = 3;
          break;

        case '4':
          number = 4;
          break;

        case '5':
          number = 5;
          break;

        case '6':
          number = 6;
          break;

        case '7':
          number = 7;
          break;

        case '8':
          number = 8;
          break;

        case '9':
          number = 9;
          break;

        case 'A':
        case 'a':
          number = 10;
          break;

        case 'B':
        case 'b':
          number = 11;
          break;

        case 'C':
        case 'c':
          number = 12;
          break;

        case 'D':
        case 'd':
          number = 13;
          break;

        case 'E':
        case 'e':
          number = 14;
          break;

        case 'F':
        case 'f':
          number = 15;
          break;

        default:
          // this number is not an ex
          return -1;
        }
        if(number > 0 )
        {
          decimal = decimal + (number * fast_power_of_16(power++));
        }
      }
      return decimal;
    }

    /// <summary>
    /// Copy the source string to the destination string
    /// </summary>
    /// <param name="source"></param>
    /// <param name="destination"></param>
    /// <param name="max_length"></param>
    /// <returns></returns>
    static void copy_string(const TJCHAR* source, TJCHAR* destination, unsigned int max_length)
    {
      if (nullptr == source)
      {
        if (destination != nullptr && max_length >= 1)
        {
          destination[0] = TJ_NULL_TERMINATOR;
        }
        return;
      }
      for (unsigned int i = 0;; ++i)
      {
        if (source[i] == TJ_NULL_TERMINATOR)
        {
          destination[i] = TJ_NULL_TERMINATOR;
          return;
        }
        if (i > max_length)
        {
          return;
        }
        if (i == max_length)
        {
          // the caller must leave space for the null terminator
          // so we assume that the actual buffer size if max_length+1
          destination[i] = TJ_NULL_TERMINATOR;
          return;
        }
        destination[i] = source[i];
      }
    }


    /// <summary>
    /// Custom case compare that probably will not work with
    /// locals and so on.
    /// </summary>
    /// <param name="s1"></param>
    /// <param name="s2"></param>
    /// <returns></returns>
    static int case_compare(const TJCHAR* s1, const TJCHAR* s2, bool case_sensitive)
    {
      if (true == case_sensitive)
      {
        return strcmp(s1, s2);
      }

      while (tolower(*s1) == tolower(*s2))
      {
        if (*s1++ == TJ_NULL_TERMINATOR)
        {
          return 0;
        }
        ++s2;
      }
      return tolower(*s1) - tolower(*s2);
    }

    /// <summary>
    /// Compare if 2 strings are the same
    /// </summary>
    /// <param name="lhs"></param>
    /// <param name="rhs"></param>
    /// <returns></returns>
    static bool are_same(const TJCHAR* lhs, const TJCHAR* rhs, bool case_sensitive)
    {
      if (nullptr == lhs && nullptr == rhs)
      {
        return true;
      }
      else if (nullptr == lhs || nullptr == rhs)
      {
        return false;
      }
      return case_compare(lhs, rhs, case_sensitive) == 0;
    }

    static void free_members(TJDICTIONARY* members)
    {
      if (members == nullptr)
      {
        return;
      }
#if TJ_INCLUDE_STDVECTOR == 1
      for (auto var : *members)
      {
        delete var;
      }
#endif
      delete members;
      members = nullptr;
    }

    static void free_values(TJLIST* values)
    {
      if (values == nullptr)
      {
        return;
      }

#if TJ_INCLUDE_STDVECTOR == 1
      for (auto var : *values)
      {
        delete var;
      }
#endif
      delete values;
      values = nullptr;
    }

    /// <summary>
    /// Fast(ish) Calculate the number of digits in a whole unsigned number.
    /// </summary>
    /// <param name="number"></param>
    /// <returns></returns>
    static unsigned long long get_number_of_digits(const unsigned long long& number)
    {
      if (number == 0)
      {
        return 0ull;
      }
      unsigned long long digit = 0;
      for (unsigned long long i = 0; i <= number; ++digit)
      {
        //  multiply by 10
        i = i == 0 ? 10 : (i << 3) + (i << 1);
      }
      return digit;
    }

    /// <summary>
    /// Increase the size of a string to make space.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="length"></param>
    /// <param name="resize_length"></param>
    /// <param name="needed_length">How much we should grow at least by</param>
    /// <returns></returns>
    static unsigned int resize_string(TJCHAR*& source, unsigned int current_length, const unsigned int needed_length)
    {
      //  create the new string
      if (source == nullptr)
      {
        auto grow_by = needed_length > TJ_DEFAULT_STRING_READ_SIZE ? needed_length : TJ_DEFAULT_STRING_READ_SIZE;
        TJCHAR* new_string = new TJCHAR[grow_by];
        memset(new_string, TJ_NULL_TERMINATOR, sizeof(TJCHAR) * grow_by);
        source = new_string;
        return grow_by;
      }

      if (current_length >= TJ_DEFAULT_STRING_MAX_READ_SIZE)
      {
        // we have reached the limit
        // we simply cannot go further and we do not want to risk further corruption.
        _Exit(-1);
      }
      unsigned int resize_length = 0;
      if (current_length >= TJ_DEFAULT_STRING_MAX_READ_GROW)
      {
        // we are about to reach the limit, if we cannot make it here, we will break.
        resize_length = TJ_DEFAULT_STRING_MAX_READ_SIZE;
      }
      else
      {
        // we are about to multiply by 2
        // so if the length is less what we need this will be an issue.
        if (current_length < needed_length)
        {
          if (needed_length > TJ_DEFAULT_STRING_MAX_READ_GROW)
          {
            resize_length += TJ_DEFAULT_STRING_MAX_READ_GROW;
          }
          else
          {
            resize_length += needed_length;
          }
        }
        else
        {
          //  multiply our capacity by 2
          resize_length = current_length << 1;
        }
      }

      // create the new array.
      TJCHAR* new_string = new TJCHAR[resize_length];
      memmove(new_string, source, current_length* sizeof(TJCHAR));
      memset(new_string+ current_length, TJ_NULL_TERMINATOR, sizeof(TJCHAR) * (resize_length- current_length));
      delete[] source;
      source = new_string;
      return resize_length;
    }

    /// <summary>
    /// As the name implies, we will add a single character to an existing string.
    /// </summary>
    /// <param name="char_to_add"></param>
    /// <param name="buffer"></param>
    /// <param name="buffer_pos"></param>
    /// <param name="buffer_max_length"></param>
    static void add_char_to_string(const TJCHAR char_to_add, TJCHAR*& buffer, int& buffer_pos, int& buffer_max_length) noexcept
    {
      if (buffer_pos + 1 >= buffer_max_length)
      {
        buffer_max_length = resize_string(buffer, buffer_max_length, 1);
      }
      buffer[buffer_pos] = char_to_add;
      buffer[buffer_pos+1] = TJ_NULL_TERMINATOR;
      ++buffer_pos;
    }

    /// <summary>
    /// Add a string to the buffer string.
    /// </summary>
    /// <param name="string_to_add"></param>
    /// <param name="buffer"></param>
    /// <param name="buffer_pos"></param>
    /// <param name="buffer_max_length"></param>
    static void add_string_to_string(const TJCHAR* string_to_add, TJCHAR*& buffer, int& buffer_pos, int& buffer_max_length) noexcept
    {
      if (nullptr == string_to_add)
      {
        return;
      }

      // what we want to add
      auto length = strlen(string_to_add);
      int total_length = length + sizeof(TJCHAR);//  we need one extra for the null char
      if (buffer_pos + total_length >= buffer_max_length)
      {
        buffer_max_length = resize_string(buffer, buffer_max_length, buffer_pos+total_length);
      }
      memcpy(buffer + buffer_pos, string_to_add, length);
      buffer[buffer_pos+length] = TJ_NULL_TERMINATOR;
      buffer_pos += length;
    }

    static bool try_add_char_to_string_after_escape(const TJCHAR*& source, TJCHAR*& result, int& result_pos, int& result_max_length)
    {
      const auto& next_char = *(source + 1);
      if (next_char == TJ_NULL_TERMINATOR)
      {
        return false;
      }

      // as per RFC8259
      switch (next_char)
      {
      case TJ_ESCAPE_QUOTATION:       // "    quotation mark  U+0022
      case TJ_ESCAPE_SOLIDUS:         // /    solidus         U+002F 
      case TJ_ESCAPE_REVERSE_SOLIDUS: // \    reverse solidus U+005C
        // skip the escpape and keep the character
        source++;
        add_char_to_string(next_char, result, result_pos, result_max_length);
        return true;

      case'f':  // f    form feed       U+000C
        source++;
        add_char_to_string(TJ_ESCAPE_FORM_FEED, result, result_pos, result_max_length);
        return true;

      case 'b': // b    backspace       U+0008
        source++;
        add_char_to_string(TJ_ESCAPE_BACKSPACE, result, result_pos, result_max_length);
        return true;

      case 'n': // n    line feed       U+000A
        source++;
        add_char_to_string(TJ_ESCAPE_LINE_FEED, result, result_pos, result_max_length);
        return true;

      case 'r': // r    carriage return U+000D
        source++;
        add_char_to_string(TJ_ESCAPE_CARRIAGE_RETURN, result, result_pos, result_max_length);
        return true;

      case 't': // t    tab             U+0009
        source++;
        add_char_to_string(TJ_ESCAPE_TAB, result, result_pos, result_max_length);
        return true;

      case 'u': // /uxxxx escape
        {
          // this is the worse case scenario .. we now have to try read the next 4 characters
          // U+0000 through U+FFFF
          TJCHAR* hex = nullptr;
          int buffer_pos = 0;
          int buffer_max_length = 0;
          for (auto i = 0; i < 4; ++i)
          {
            const auto& possible_hex_char = *(source + i + 2);  //  we add two for '/u'
            switch (possible_hex_char)
            {
            TJ_CASE_HEX
              add_char_to_string(possible_hex_char, hex, buffer_pos, buffer_max_length);
              break;

            default:
            case TJ_NULL_TERMINATOR:
                //  not sure what this is, but it is not valid.
                delete[] hex;
                return false;
            }
          }

          auto decimal = fast_hex_to_decimal(hex);
          delete[] hex;
          if (decimal < 0)
          {
            return false; //  not sure what this is.
          }
#if TJ_USE_CHAR == 1
          if (decimal <= 0x7F) 
          {
            // 1-byte UTF-8 (ASCII)
            add_char_to_string(static_cast<char>(decimal), result, result_pos, result_max_length);
          }
          else if (decimal <= 0x7FF) 
          {
            // 2-byte UTF-8
            add_char_to_string(static_cast<char>(0xC0 | (decimal >> 6)), result, result_pos, result_max_length);
            add_char_to_string(static_cast<char>(0x80 | (decimal & 0x3F)), result, result_pos, result_max_length);
          }
          else if (decimal <= 0xFFFF)
          {
            // 3-byte UTF-8
            add_char_to_string(static_cast<char>(0xE0 | (decimal >> 12)), result, result_pos, result_max_length);
            add_char_to_string(static_cast<char>(0x80 | ((decimal >> 6) & 0x3F)), result, result_pos, result_max_length);
            add_char_to_string(static_cast<char>(0x80 | (decimal & 0x3F)), result, result_pos, result_max_length);
          }
#else
          add_char_to_string(static_cast<TJCHAR>(decimal), result, result_pos, result_max_length);
#endif
          source += 6;  //  the full \uXXXX = 6 char
          return true;
        }
        return false;
      }

      //  this is not an escaped character, just a single reverse solidus
      return false;
    }

    static TJCHAR* try_continue_read_string(const TJCHAR*& p, ParseResult& parse_result)
    {
      int result_pos = 0;
      int result_max_length = 0;
      TJCHAR* result = nullptr;
      while (*p != TJ_NULL_TERMINATOR)
      {
        switch (*p)
        {
        TJ_CASE_SPACE
          //  only read if we started.
          switch (*p)
          {
          case TJ_ESCAPE_LINE_FEED:       // % x6E / ; n    line feed       U + 000A
          case TJ_ESCAPE_CARRIAGE_RETURN: // % x72 / ; r    carriage return U + 000D
          case  TJ_ESCAPE_TAB:            // % x74 / ; t    tab             U + 0009
            // ERROR: invalid character inside the string.
            delete[] result;
            parse_result.assign_exception_message("Invalid character inside the string.");
            return nullptr;
          }
          add_char_to_string(*p, result, result_pos, result_max_length);
          p++;
          break;

        TJ_CASE_MAYBE_ESCAPE
          if (!try_add_char_to_string_after_escape(p, result, result_pos, result_max_length))
          {
            delete[] result;
            // ERROR: invalid/unknown character after single reverse solidus.
            parse_result.assign_exception_message("Invalid/unknown character after single reverse solidus.");
            return nullptr;
          }
          p++;
          break;

        case TJ_ESCAPE_SOLIDUS:         // % x2F / ; /    solidus         U + 002F
          // solidus can be escaped ... and not escaped...
          // this is a case where it is not escaped.
          add_char_to_string(*p, result, result_pos, result_max_length);
          p++;
          break;

        case TJ_ESCAPE_BACKSPACE:       // % x62 / ; b    backspace       U + 0008
        case TJ_ESCAPE_FORM_FEED:       // % x66 / ; f    form feed       U + 000C
          delete[] result;
          // ERROR: invalid character inside the string.
          parse_result.assign_exception_message("Invalid character inside the string..");
          return nullptr;

        TJ_CASE_START_STRING
          p++;

          // Allocate memory for the result string
          // Null-terminate the string
          add_char_to_string(TJ_NULL_TERMINATOR, result, result_pos, result_max_length);
          return result;

        default:
          // we are still in the string, then we are good.
          add_char_to_string(*p, result, result_pos, result_max_length);
          p++;
          break;
        }
      }

      // // ERROR: we could not close the object.
      delete[] result;
      parse_result.assign_exception_message("We could not close the string.");
      return nullptr;
    }

    static bool try_skip_colon(const TJCHAR*& p)
    {
      while (*p != TJ_NULL_TERMINATOR)
      {
        switch (*p)
        {
          TJ_CASE_SPACE
            p++;
          break;

          TJ_CASE_COLON
            p++;
          return true;

        default:
          // ERROR: could not find the expected colon
          return false;
        }
      }

      // ERROR: This should never be reached, unless the string does not contain a '\0'
      return false;
    }

    static TJValue* try_continue_read_true(const TJCHAR*& p)
    {
      if (*(p) != 'r')
      {
        return nullptr;
      }
      if (*(p + 1) != 'u')
      {
        return nullptr;
      }
      if (*(p + 2) != 'e')
      {
        return nullptr;
      }

      p += 3;

      // all good.
      return new TJValueBoolean(true);
    }

    static TJValue* try_continue_read_false(const TJCHAR*& p)
    {
      if (*(p) != 'a')
      {
        return nullptr;
      }
      if (*(p + 1) != 'l')
      {
        return nullptr;
      }
      if (*(p + 2) != 's')
      {
        return nullptr;
      }
      if (*(p + 3) != 'e')
      {
        return nullptr;
      }

      p += 4;

      // all good.
      return new TJValueBoolean(false);
    }

    static TJValue* try_continue_read_null(const TJCHAR*& p)
    {
      if (*(p) != 'u')
      {
        return nullptr;
      }
      if (*(p + 1) != 'l')
      {
        return nullptr;
      }
      if (*(p + 2) != 'l')
      {
        return nullptr;
      }

      p += 3;

      // all good.
      return new TJValueNull();
    }

    static TJValue* try_create_number_from_float(long double value)
    {
      auto is_negative = false;
      if (value < 0)
      {
        value = std::abs(value);
        is_negative = true;
      }
      auto initial_pos_value = value;
      long double int_part;
      long double frac_part = std::modf(value, &int_part);
      long long whole = static_cast<long long>(int_part);

      int decimal_digits = 0;
      while (std::abs(frac_part) > std::numeric_limits<long double>::epsilon() && decimal_digits < TJ_MAX_NUMBER_OF_DIGGITS)
      {
        value *= 10;
        frac_part = std::modf(value, &int_part);
        ++decimal_digits;
      }

      // Shift the fractional part to preserve `precision` decimal digits
      long double scaled_frac = std::modf(initial_pos_value, &int_part) * std::pow(10.0L, decimal_digits);
      long long fraction = static_cast<long long>(scaled_frac);
      if (fraction == 0)
      {
        return new TJValueNumberInt(is_negative ? -1 * whole : whole);
      }
      return new TJValueNumberFloat(whole, fraction, decimal_digits, is_negative);
    }

    static unsigned int get_unsigned_exponent_from_float(long double value)
    {
      value = std::abs(value);
      long double int_part;
      long double frac_part = std::modf(value, &int_part);

      int decimal_digits = 0;
      while (std::abs(frac_part) > std::numeric_limits<long double>::epsilon() && decimal_digits < TJ_MAX_NUMBER_OF_DIGGITS)
      {
        value *= 10;
        frac_part = std::modf(value, &int_part);
        ++decimal_digits;
      }
      return decimal_digits;
    }

    static unsigned long long get_fraction_from_float(long double value)
    {
      value = std::abs(value);
      auto initial_pos_value = value;
      long double int_part;
      long double frac_part = std::modf(value, &int_part);

      int decimal_digits = 0;
      while (std::abs(frac_part) > std::numeric_limits<long double>::epsilon() && decimal_digits < TJ_MAX_NUMBER_OF_DIGGITS)
      {
        value *= 10;
        frac_part = std::modf(value, &int_part);
        ++decimal_digits;
      }

      // Shift the fractional part to preserve `precision` decimal digits
      long double scaled_frac = std::modf(initial_pos_value, &int_part) * std::pow(10.0L, decimal_digits);
      return static_cast<unsigned long long>(scaled_frac);
    }

    static unsigned long long get_whole_number_from_float(long double value)
    {
      value = std::abs(value);
      long double int_part;
      std::modf(value, &int_part);
      // we know it is not negative
      return static_cast<unsigned long long>(int_part);
    }

    static TJValue* try_create_number_from_parts_no_exponent(const bool& is_negative, const unsigned long long& unsigned_whole_number, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent)
    {
      if (unsigned_fraction == 0)
      {
        // zero is a positive number
        return new TJValueNumberInt(unsigned_whole_number, unsigned_whole_number == 0 ? false : is_negative);
      }
      return new TJValueNumberFloat(unsigned_whole_number, unsigned_fraction, fraction_exponent, is_negative);
    }

    static unsigned long long shift_number_left(const unsigned long long source, const unsigned long long exponent)
    {
      if (exponent == 0)
      {
        return source;
      }
      const auto muliplier = fast_power_of_10(exponent);
      return source * muliplier;
    }

    static unsigned long long shift_number_right(const unsigned long long source, const unsigned long long exponent, unsigned long long& shifted_source)
    {
      const auto divider = fast_power_of_10(exponent);
      auto new_source = static_cast<unsigned long long>(source / divider);
      shifted_source = source - new_source * divider;
      return new_source;
    }

    static unsigned long long shift_fraction_left(const unsigned long long& fraction, const unsigned long long& fraction_exponent, const unsigned long long& exponent, unsigned long long& shifted_fraction, unsigned long long& shitfed_unsigned_fraction_exponent)
    {
      if (exponent > fraction_exponent)
      {
        // we are moving more to the left than we have fractions
        // so we just need to move the extra fraction
        shifted_fraction = 0;
        shitfed_unsigned_fraction_exponent = 0;
        return shift_number_left(fraction, exponent - fraction_exponent);
      }

      if (exponent == fraction_exponent)
      {
        // no shifting needed the number is already what we need.
        shifted_fraction = 0;
        shitfed_unsigned_fraction_exponent = 0;
        return fraction;
      }

      shitfed_unsigned_fraction_exponent = fraction_exponent - exponent;

      // we know that the fraction_exponent is bigger than the exponent.
      // so we are not shifting the whole way but we have to be careful as the
      // len of the fraction might actually be less than the fraction because of leading 0s
      // for example 0.0012 and 0.12 have a len of 2 but a fraction_exponent of 2 and 4
      // the length can never be more than the fraction exponent.
      const auto& fraction_length = get_number_of_digits(fraction);

      if (fraction_length == fraction_exponent)
      {
        const auto& divider = fast_power_of_10(shitfed_unsigned_fraction_exponent);
        const auto& shifted_unsigned_fraction = static_cast<unsigned long long>(fraction / divider);
        shifted_fraction = fraction - static_cast<unsigned long long>(shifted_unsigned_fraction * divider);
        return shifted_unsigned_fraction;
      }

      if (fraction_exponent - fraction_length <= 0)
      {
        // the number is 0.0012 and we want to shift 2 or less
        // so the new faction is 0.12 and the return number is zero
        shifted_fraction = fraction;
        return 0ll;
      }

      // the number is 0.0123 and we want to shift 3
      //   the retrun number is 12, (0123 shifted leftx3)
      //   the return fraction is 3, (0123 shifted leftx3 - 12)
      //   the return fraction_exponent is 1, (0123 shifted leftx3 - 12)
      // The number of leading zeros, (that we have to ignore), is the fraction_exponent - fraction_length
      const auto& leading_zeros = fraction_exponent - fraction_length;
      if (leading_zeros >= exponent)
      {
        // we have more leading zeros than the number of exponents we are trying to shift.
        // so the fraction remains the same and the shitfed_unsigned_fraction_exponent has already been updated.
        shifted_fraction = fraction;
        return 0ll;
      }

      auto divider = fast_power_of_10(shitfed_unsigned_fraction_exponent);
      const auto& shifted_unsigned_fraction = static_cast<unsigned long long>(fraction / divider);
      shifted_fraction = fraction - static_cast<unsigned long long>(shifted_unsigned_fraction * divider);
      return shifted_unsigned_fraction;
    }

    static TJValue* try_create_number_from_parts_positive_exponent_no_whole_number(const bool& is_negative, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent, const unsigned long long& exponent)
    {
      if (exponent >= fraction_exponent)
      {
        unsigned long long shifted_unsigned_fraction = 0;
        unsigned long long shitfed_unsigned_fraction_exponent = 0;
        const auto& fraction_length = get_number_of_digits(unsigned_fraction);
        const auto& leading_zeros = fraction_exponent - fraction_length;
        // we just want the first number so we are passing a 1x exponent only
        // but we need to add the number of leading zeros to make sure that we shift properly.
        const auto& shifted_unsigned_whole_number = shift_fraction_left(unsigned_fraction, fraction_exponent, leading_zeros + 1, shifted_unsigned_fraction, shitfed_unsigned_fraction_exponent);

        const auto& shifted_fraction_exponent = exponent - fraction_exponent;
        if (shifted_fraction_exponent <= TJ_MAX_NUMBER_OF_DIGGITS)
        {
          if (shifted_unsigned_fraction == 0)
          {
            return new TJValueNumberInt(shift_number_left(shifted_unsigned_whole_number, shifted_fraction_exponent), is_negative);
          }

          return new TJValueNumberFloat(
            shift_number_left(shifted_unsigned_whole_number, shifted_fraction_exponent),
            shifted_unsigned_fraction,
            shifted_fraction_exponent,
            is_negative);
        }

        // TODO: Cases where exponent is > than TJ_MAX_NUMBER_OF_DIGGITS
        return nullptr;
      }


      // the number is something like 0.00001e+3 the fraction_exponent is 4 and the exponent is 3
      // so we can just move the fraction to the left the whole number will remain zero
      const auto& shifted_fraction_exponent = fraction_exponent - exponent;
      if (shifted_fraction_exponent <= TJ_MAX_NUMBER_OF_DIGGITS)
      {
        // the number cannot be an int as it would mean that both
        // the whole number and the fraction are zer0
        return new TJValueNumberFloat(0ull, unsigned_fraction, shifted_fraction_exponent, is_negative);
      }

      // TODO: Cases where exponent is > than TJ_MAX_NUMBER_OF_DIGGITS
      return nullptr;
    }

    static TJValue* try_create_number_from_parts_positive_exponent(const bool& is_negative, const unsigned long long& unsigned_whole_number, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent, const unsigned long long& exponent)
    {
      const auto number_of_digit_whole = get_number_of_digits(unsigned_whole_number);

      // case 1:
      //   The total number is less than TJ_MAX_NUMBER_OF_DIGGITS
      //   so we can get rid of the exponent altogether.
      if (fraction_exponent <= exponent && number_of_digit_whole + exponent <= TJ_MAX_NUMBER_OF_DIGGITS)
      {
        // we know that the fraction will disapear because it is smaller than the total fraction
        // we want to first move the whole number by the number of fractions
        auto shifted_unsigned_whole_number = shift_number_left(unsigned_whole_number, fraction_exponent);
        // then add the fraction
        shifted_unsigned_whole_number += unsigned_fraction;
        // then shift it again with the rest of the exponent
        shifted_unsigned_whole_number = shift_number_left(shifted_unsigned_whole_number, exponent - fraction_exponent);

        return new TJValueNumberInt(shifted_unsigned_whole_number, is_negative);
      }

      if (fraction_exponent > exponent && (number_of_digit_whole + exponent) <= TJ_MAX_NUMBER_OF_DIGGITS)
      {
        // we now know that the fraction will not completely shift.
        // so we must move the whole number by the number of expoent
        auto shifted_unsigned_whole_number = shift_number_left(unsigned_whole_number, exponent);

        // we then want to shift the fraction by the number of exponent and add that to the list.
        unsigned long long shifted_unsigned_fraction_exponent = 0;
        unsigned long long shifted_unsigned_fraction = 0;
        shifted_unsigned_whole_number += shift_fraction_left(unsigned_fraction, fraction_exponent, exponent, shifted_unsigned_fraction, shifted_unsigned_fraction_exponent);

        // as we sifted the fraction by the number of exponent
        // then the size of the fraction is smaller by the exponent.
        return new TJValueNumberFloat(shifted_unsigned_whole_number, shifted_unsigned_fraction, shifted_unsigned_fraction_exponent, is_negative);
      }

      // case 2:
      //  The total number if more than TJ_MAX_NUMBER_OF_DIGGITS
      //  So we have to move the whole number to be a single digit.
      //  and the fraction to be shifted accordingly
      //  and the exponent moved.

      // case 2a:
      //  The whole number if zero ... in that case we have to shift the fraction to the first whole number.
      //  But we might not make it and in that case the whole number will remain to zero
      //  But the fraction will shift one way or the other.
      if (unsigned_whole_number == 0)
      {
        return try_create_number_from_parts_positive_exponent_no_whole_number(is_negative, unsigned_fraction, fraction_exponent, exponent);
      }

      // case 2b:
      //   The whole number is more than zero _and_ the fraction is also non zero.
      //   The total is, (currently), greater than TJ_MAX_NUMBER_OF_DIGGITS
      const unsigned long long shifted_unsigned_whole_number_exponent = number_of_digit_whole - 1;
      unsigned long long shifted_unsigned_fraction = 0;
      const auto& shifted_unsigned_whole_number = shift_number_right(unsigned_whole_number, shifted_unsigned_whole_number_exponent, shifted_unsigned_fraction);

      // we then need to add shifted_unsigned_fraction in front of unsigned_fraction
      auto shifted_fraction_exponent = shifted_unsigned_whole_number_exponent + (fraction_exponent - shifted_unsigned_whole_number_exponent);
      shifted_unsigned_fraction = (shifted_unsigned_fraction * fast_power_of_10(shifted_fraction_exponent)) + unsigned_fraction;

      // and the exponent also shitt byt the number we moved.
      const unsigned long long shifted_exponent = exponent + shifted_unsigned_whole_number_exponent;

      return new TJValueNumberExponent(
        shifted_unsigned_whole_number,
        shifted_unsigned_fraction,
        (shifted_unsigned_whole_number_exponent + fraction_exponent),
        shifted_exponent,
        is_negative);
    }

    static TJValue* try_create_number_from_parts_negative_exponent(const bool& is_negative, const unsigned long long& unsigned_whole_number, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent, const unsigned long long& exponent)
    {
      // if the number is something like 123.456 with e=2
      // then the number will become 12345.6 e=0
      // so we need the number of digits.
      const auto& number_of_digit_whole = get_number_of_digits(unsigned_whole_number);
      const auto& number_of_digit_fraction = get_number_of_digits(unsigned_fraction);

      // case 1:
      //   The total number is less than TJ_MAX_NUMBER_OF_DIGGITS
      //   so we can get rid of the exponent altogether.
      if (number_of_digit_whole + number_of_digit_fraction + exponent <= TJ_MAX_NUMBER_OF_DIGGITS)
      {
        // we will shift the whole number to the left right by the number of exponents.
        // so 2e-3 would become 0.002
        // then we wil shift the number of fraction to the right by the number if exponent.
        // we will then add the two together.
        unsigned long long shifted_unsigned_fraction = 0;
        const auto& shifted_unsigned_whole_number = shift_number_right(
          unsigned_whole_number, 
          exponent, 
          shifted_unsigned_fraction);

        // we then need to shift the faction iseft to the right a little so we can add the given fraction.
        shifted_unsigned_fraction = shift_number_left(shifted_unsigned_fraction, fraction_exponent);
        shifted_unsigned_fraction += unsigned_fraction;

        if (shifted_unsigned_fraction == 0)
        {
          return new TJValueNumberInt(shifted_unsigned_whole_number, is_negative);
        }

        const auto& shifted_fraction_exponent = fraction_exponent + exponent;
        return new TJValueNumberFloat(shifted_unsigned_whole_number, shifted_unsigned_fraction, shifted_fraction_exponent, is_negative);
      }

      // case 2:
      //  The total number if more than TJ_MAX_NUMBER_OF_DIGGITS
      //  So we have to move the whole number to be a single digit.
      //  and the fraction to be shifted accordingly
      //  and the exponent moved.

      // case 2a:
      //  The whole number if zero ... in that case we have to shift the fraction to the first whole number.
      //  But we might not make it and in that case the whole number will remain to zero
      //  But the fraction will shift one way or the other.
      if (unsigned_whole_number == 0)
      {
        return try_create_number_from_parts_negative_exponent_no_whole_number(is_negative, unsigned_fraction, fraction_exponent, exponent);
      }

      // case 2b:
      //   The whole number is more than zero _and_ the fraction is also non zero.
      //   The total is, (currently), greater than TJ_MAX_NUMBER_OF_DIGGITS
      const unsigned long long shifted_unsigned_whole_number_exponent = number_of_digit_whole - 1;
      unsigned long long shifted_unsigned_fraction = 0;
      const auto& shifted_unsigned_whole_number = shift_number_right(unsigned_whole_number, shifted_unsigned_whole_number_exponent, shifted_unsigned_fraction);

      // we then need to add shifted_unsigned_fraction in front of unsigned_fraction
      auto shifted_fraction_exponent = shifted_unsigned_whole_number_exponent + (fraction_exponent - shifted_unsigned_whole_number_exponent);
      shifted_unsigned_fraction = (shifted_unsigned_fraction * fast_power_of_10(shifted_fraction_exponent)) + unsigned_fraction;

      // and the exponent also shitt by the number we moved.
      // as it is a negative exponent we need to move to the left.
      const unsigned long long shifted_exponent = exponent - shifted_unsigned_whole_number_exponent;

      return new TJValueNumberExponent(
        shifted_unsigned_whole_number,
        shifted_unsigned_fraction,
        (shifted_unsigned_whole_number_exponent + fraction_exponent),
        -1 * shifted_exponent,
        is_negative);
    }

    static TJValue* try_create_number_from_parts_negative_exponent_no_whole_number(const bool& is_negative, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent, const unsigned long long& exponent)
    {
      //
      // remember that this is a negative exponent ...
      //

      // if we have a fraction and no whole number then we can move the number to the right
      unsigned long long shifted_unsigned_fraction_exponent = 0;
      unsigned long long shifted_unsigned_fraction = 0;
      const auto& fraction_length = get_number_of_digits(unsigned_fraction);
      const auto& leading_zeros = fraction_exponent - fraction_length;
      // we just want the first number so we are passing a 1x exponent only
      // but we need to add the number of leading zeros to make sure that we shift properly.
      const auto& shifted_unsigned_whole_number = shift_fraction_left(unsigned_fraction, fraction_exponent, leading_zeros + 1, shifted_unsigned_fraction, shifted_unsigned_fraction_exponent);

      const auto& actual_shifted_fraction_exponent = exponent + (fraction_exponent - shifted_unsigned_fraction_exponent);

      if (actual_shifted_fraction_exponent <= TJ_MAX_NUMBER_OF_DIGGITS)
      {
        if (shifted_unsigned_fraction == 0)
        {
          return new TJValueNumberInt(shift_number_left(shifted_unsigned_whole_number, shifted_unsigned_fraction_exponent), is_negative);
        }
        return new TJValueNumberFloat(
          shift_number_left(shifted_unsigned_whole_number, shifted_unsigned_fraction_exponent),
          shifted_unsigned_fraction,
          shifted_unsigned_fraction_exponent,
          is_negative);
      }

      // TODO: Cases where exponent is > than TJ_MAX_NUMBER_OF_DIGGITS
      return new TJValueNumberExponent(
        shifted_unsigned_whole_number,
        shifted_unsigned_fraction,
        shifted_unsigned_fraction_exponent,
        -1 * actual_shifted_fraction_exponent,
        is_negative);
    }

    static TJValue* try_create_number_from_parts(const bool& is_negative, const unsigned long long& unsigned_whole_number, const unsigned long long& unsigned_fraction, const unsigned int& fraction_exponent, const long long& exponent)
    {
      // no exponent number is int or float
      if (exponent == 0)
      {
        return try_create_number_from_parts_no_exponent(is_negative, unsigned_whole_number, unsigned_fraction, fraction_exponent);
      }

      // positive exponent.
      if (exponent > 0)
      {
        return try_create_number_from_parts_positive_exponent(is_negative, unsigned_whole_number, unsigned_fraction, fraction_exponent, exponent);
      }

      // the exponent is negative, so we need to either shift the whole number and the fraction
      // but we have to be careful how it is shifted so we do not overflow one way or another.
      const auto& positive_exponent = -1 * exponent;
      return try_create_number_from_parts_negative_exponent(is_negative, unsigned_whole_number, unsigned_fraction, fraction_exponent, positive_exponent);
    }

    /// <summary>
    /// The function looks for a whole number and stops as soon as we find decimal and/or exponent.
    /// </summary>
    /// <param name="p">The current string pointer.</param>
    /// <returns></returns>
    static TJCHAR* try_read_whole_number(const TJCHAR*& p, ParseResult& parse_result)
    {
      const TJCHAR* start = nullptr;
      int found_spaces = 0;
      while (*p != TJ_NULL_TERMINATOR)
      {
        switch (*p)
        {
          TJ_CASE_SPACE
            if (nullptr != start)
            {
              ++found_spaces;
            }
            p++;
            break;

          TJ_CASE_DIGIT
            if (nullptr == start)
            {
              start = p; // this is the start
            }
            if (found_spaces > 0)
            {
              // ERROR: Number has a space between it.
              parse_result.assign_exception_message("Number has a space between it.");
              return nullptr;
            }
            p++;
            break;

        default:
          return read_string( start, p, found_spaces);
        }
      }

      return read_string(start, p, found_spaces);
    }

    static TJCHAR* read_string(const TJCHAR* start, const TJCHAR* end, int spaces)
    {
      if (nullptr == start)
      {
        // ERROR: unknown character
        return nullptr;
      }
      // Calculate the length of the text inside the quotes
      const auto& length = end - start - spaces;
      // Allocate memory for the result string
      TJCHAR* result = new TJCHAR[length + 1];
      TJHelper::copy_string(start, result, length);
      result[length] = TJ_NULL_TERMINATOR; // Null-terminate the string
      return result;
    }

    static TJCHAR* try_read_whole_number_as_fraction(const TJCHAR*& p, ParseResult& parse_result)
    {
      // try read the number
      auto whole_number = try_read_whole_number(p, parse_result);
      if (nullptr == whole_number)
      {
        parse_result.assign_exception_message("Fraction does not have a number after the token '.'");
        return nullptr;
      }
      // trip the trailling zeros are they are not needed in a fraction
      // but we need to make sure that we have at least one
      // so 42.000000 becomes 42.0
      auto len = TJHelper::string_length(whole_number);
      while (len > 1 && whole_number[len - 1] == '0')
      {
        whole_number[len - 1] = TJ_NULL_TERMINATOR;
        --len;
      }
      return whole_number;
    }

    static unsigned long long fast_string_to_long_long(const TJCHAR* p)
    {
      // the numbers are unsigned and should only contain digits.
      // so we do not have signs or letters to worry about.
      long long result = 0;
      while (*p != TJ_NULL_TERMINATOR)
      {
        TJCHAR c = *p;
        switch (c)
        {
          // it might sound silly to do it that way but it is faster
          // than doing something like result += c - '0'
        case '0':
          result = TJHelper::fast_multiply_by_10(result);
          p++;
          break;

        case '1':
          result = TJHelper::fast_multiply_by_10(result);
          result += 1;
          p++;
          break;

        case '2':
          result = TJHelper::fast_multiply_by_10(result);
          result += 2;
          p++;
          break;

        case '3':
          result = TJHelper::fast_multiply_by_10(result);
          result += 3;
          p++;
          break;

        case '4':
          result = TJHelper::fast_multiply_by_10(result);
          result += 4;
          p++;
          break;

        case '5':
          result = TJHelper::fast_multiply_by_10(result);
          result += 5;
          p++;
          break;

        case '6':
          result = TJHelper::fast_multiply_by_10(result);
          result += 6;
          p++;
          break;

        case '7':
          result = TJHelper::fast_multiply_by_10(result);
          result += 7;
          p++;
          break;

        case '8':
          result = TJHelper::fast_multiply_by_10(result);
          result += 8;
          p++;
          break;

        case '9':
          result = TJHelper::fast_multiply_by_10(result);
          result += 9;
          p++;
          break;

        default:
          // ERROR: only numbers are expected.
          return 0ull;
        }
      }
      return result;
    }

    static bool has_possible_double_zero(const TJCHAR* p)
    {
      if (p[0] == TJ_NULL_TERMINATOR || p[1] == TJ_NULL_TERMINATOR || p[0] != '0')
      {
        return false;
      }
      // if the number is 0121 then it is wrong
      // but if we have 0.121 then it is valid
      return p[0] == '0' && p[1] != '.';
    }

    /// <summary>
    /// Try and read a number given a string.
    /// </summary>
    /// <param name="p"></param>
    /// <param name="parse_result"></param>
    /// <returns></returns>
    static TJValue* try_read_number(const TJCHAR*& p, ParseResult& parse_result)
    {
      bool is_negative = false;
      if (*p == '-')
      {
        is_negative = true;
        p++;
      }

      // then try and read the digit(s).
      auto possible_number = try_read_whole_number(p, parse_result);
      if (nullptr == possible_number)
      {
        // ERROR: Could not locate the number.
        return nullptr;
      }

      if (has_possible_double_zero(possible_number))
      {
        // ERROR: Numbers cannot have leading zeros
        parse_result.assign_exception_message("Numbers cannot have leading zeros.");
        delete[] possible_number;
        return nullptr;
      }

      // convert that number to an unsigned long, long
      const auto& unsigned_whole_number = fast_string_to_long_long(possible_number);
      delete[] possible_number;

      // read the faction if there is one.
      unsigned long long unsigned_fraction = 0;
      unsigned int fraction_exponent = 0;
      if (*p == '.')
      {
        p++;
        const auto& possible_fraction_number = try_read_whole_number_as_fraction(p, parse_result);
        if (nullptr == possible_fraction_number)
        {
          // ERROR: we cannot have a number like '-12.' or '42.
          return nullptr;
        }

        // so 001 become exponent = 3
        fraction_exponent = string_length(possible_fraction_number);
        unsigned_fraction = fast_string_to_long_long(possible_fraction_number);
        delete[] possible_fraction_number;
      }

      // try read the exponent if there is one.
      long long exponent = 0;
      if (*p == 'e' || *p == 'E')
      {
        p++;

        bool is_negative_exponent = false;
        if (*p == '-')
        {
          is_negative_exponent = true;
          p++;
        }
        if (*p == '+')
        {
          is_negative_exponent = false;
          p++;
        }
        const auto& possible_exponent = try_read_whole_number(p, parse_result);
        if (nullptr == possible_exponent)
        {
          // ERROR: we cannot have a number like '-12e' or '42e
          parse_result.assign_exception_message("Number has exponent 'e' or 'E' but does not have a number.");
          return nullptr;
        }

        const auto& unsigned_exponent = fast_string_to_long_long(possible_exponent);
        delete[] possible_exponent;

        // as per the spec it is allowed to have 1e00
        // https://github.com/FFMG/tinyjson/issues/14
        if (0 == unsigned_exponent)
        {
          exponent = 0;
          is_negative_exponent = false;
        }
        else
        {
          exponent = is_negative_exponent ? unsigned_exponent * -1 : unsigned_exponent;
        }
      }
      return try_create_number_from_parts(is_negative, unsigned_whole_number, unsigned_fraction, fraction_exponent, exponent);
    }

    /// <summary>
    /// We are moving the owership of the TJMember to the array.
    /// If the array is not created we will create it and add the value.
    /// Duplicate values will be overwiten here, the old value will be removed and the new one added.
    /// </summary>
    /// <param name="member"></param>
    /// <param name="members"></param>
    static void move_member_to_members(TJMember* member, TJDICTIONARY*& members)
    {
      if (nullptr == members)
      {
        members = new TJDICTIONARY();
      }
#if TJ_INCLUDE_STDVECTOR == 1
      else
      {
        auto current = std::find_if(members->begin(), members->end(), [&](TJMember*& elem) 
          { 
            return TJHelper::are_same(elem->name(), member->name(), true);
          });
        if (current != members->end())
        {
          delete* current;
          *current = member;
          return;
        }
      }
      members->push_back(member);
#else
      members->set(member);
#endif
    }

    /// <summary>
    /// Try and read an object after we have read the openning bracket
    /// This is to prevent having to read the same char more than once.
    /// </summary>
    /// <param name="p"></param>
    /// <returns></returns>
    static TJValue* try_continue_read_object(const TJCHAR*& p, ParseResult& parse_result)
    {
      if (parse_result.current_depth() >= parse_result.options().max_depth)
      {
        // Error: We reached the max depth
        parse_result.assign_exception_message("Reached the max parse depth (object).");
        return nullptr;
      }
      //  assume no members in that object.
      bool found_comma = false;
      TJDICTIONARY* members = nullptr;
      bool after_string = false;
      bool waiting_for_a_string = false;
      while (*p != TJ_NULL_TERMINATOR)
      {
        TJCHAR c = *p;
        switch (c)
        {
        TJ_CASE_SPACE
          p++;
          break;

        TJ_CASE_END_OBJECT
          // but is it what we expected?
          if (waiting_for_a_string)
          {
            // ERROR: unexpected end of object, there was a "," after
            //        the last string and we expected a string now, not a close "}"
            free_members(members);
            parse_result.assign_exception_message("Unexpected end of object, there was a ', ' after the last string.");
            return nullptr;
          }
          p++;

          // we are done, we found it.
          // we give the ownership of the members over.
          return TJValueObject::move(members);

        TJ_CASE_START_STRING
        {
          // we got our string, no longer waiting for one.
          waiting_for_a_string = false;

          // we are no longer after the string
          after_string = false;

          // if we have members then it means we must have a comma
          // as we are expecting the elements to be separated by a comma
          // if we have no elements then it is the first one and it does not matter.
          if (members != nullptr && !found_comma)
          {
            // ERROR: expected a comma after the last element
            free_members(members);
            parse_result.assign_exception_message("Expected a comma after the last element.");
            return nullptr;
          }

          // read the actual string and value
          // that's the way it has to be.
          auto member = try_read_string_and_value(p, parse_result);
          if (member == nullptr)
          {
            // ERROR: There was an error reading the name and/or the value
            free_members(members);
            return nullptr;
          }

          found_comma = false;
          move_member_to_members(member, members);
          
          after_string = true;
        }
        break;

        TJ_CASE_COMMA
          if (!after_string)
          {
            // ERROR: found a comma out of order
            free_members(members);
            parse_result.assign_exception_message("Found a comma out of order.");
            return nullptr;
          }
          // we are no longer after the string
          after_string = false;
          waiting_for_a_string = true;
          found_comma = true;
          p++;
          break;

        default:
          // ERROR: unknown character
          free_members(members);
          parse_result.assign_exception_message("Unknown character.");
          return nullptr;
        }
      }

      // ERROR end of the string was found and we didn't find what we needed.
      free_members(members);
      parse_result.assign_exception_message("End of the string was found and we didn't find what we needed.");
      return nullptr;
    }

    /// <summary>
    /// Try and read an array after we have read the opening bracket.
    /// This is to prevent having to read the same char more than once.
    /// </summary>
    /// <param name="p"></param>
    /// <returns></returns>
    static TJValue* try_continue_read_array(const TJCHAR*& p, ParseResult& parse_result)
    {
      if (parse_result.current_depth() >= parse_result.options().max_depth)
      {
        // Error: We reached the max depth
        parse_result.assign_exception_message("Reached the max parse depth (array).");
        return nullptr;
      }

      //  assume no values in that array
      TJLIST* values = nullptr;
      bool waiting_for_a_value = true;
      bool found_comma = false;
      while (*p != TJ_NULL_TERMINATOR)
      {
        TJCHAR c = *p;
        switch (c)
        {
        TJ_CASE_SPACE
          p++;
          break;

        TJ_CASE_END_ARRAY
          if (found_comma && waiting_for_a_value)
          {
            // ERROR: unexpected end of array, there was a "," after
            //        the last value and we expected a value now, not a close "]"
            free_values(values);
            parse_result.assign_exception_message("Unexpected end of array, there was a ', ' after the last string.");
            return nullptr;
          }
          p++;

          // we are done, we found it.
          // we give the ownership of the members over.
          return TJValueArray::move(values);

        TJ_CASE_COMMA
          if (waiting_for_a_value)
          {
            // ERROR: found a comma out of order, (2 commas)
            free_values(values);
            parse_result.assign_exception_message("Found a comma out of order, (2 commas).");
            return nullptr;
          }
          // we are now waiting for a value
          waiting_for_a_value = true;
          found_comma = true;
          p++;
          break;

        default:
          const auto& value = try_read_Value(p, parse_result);
          if (value == nullptr)
          {
            // ERROR: unknown character
            free_values(values);
            return nullptr;
          }
          if (nullptr == values)
          {
            values = new TJLIST();
          }
          else if (found_comma == false && values->size() > 0)
          {
            // ERROR: We found a value but we expected a comma.
            delete value;
            free_values(values);
            parse_result.assign_exception_message("We found a value but we expected a comma.");
            return nullptr;
          }
#if TJ_INCLUDE_STDVECTOR == 1
          values->push_back(value);
#else
          values->add(value);
#endif
          waiting_for_a_value = false;
          found_comma = false;
          break;
        }
      }

      // ERROR: end of the string was found and we didn't find what we needed.
      free_values(values);
      parse_result.assign_exception_message("End of the string was found and we didn't find what we needed.");
      return nullptr;
    }

    static TJValue* try_read_Value(const TJCHAR*& p, ParseResult& parse_result)
    {
      while (*p != TJ_NULL_TERMINATOR)
      {
        TJCHAR c = *p;
        switch (c)
        {
        TJ_CASE_SPACE
          p++;
          break;

        TJ_CASE_START_STRING
        {
          auto string_value = try_continue_read_string(++p, parse_result);
          if (nullptr == string_value)
          {
            //  ERROR: could not read the string properly.
            return nullptr;
          }

          // whave read the string
          // no need to try and move further forward.
          return TJValueString::move(string_value);
        }

        case 't':
          {
            auto true_value = try_continue_read_true(++p);
            if (nullptr == true_value)
            {
              //  ERROR could not read the word 'true'
              parse_result.assign_exception_message("Could not read the word 'true'.");
              return nullptr;
            }
            return true_value;
          }

        case 'f':
        {
          auto false_value = try_continue_read_false(++p);
          if (nullptr == false_value)
          {
            //  ERROR: could not read the word 'false'
            parse_result.assign_exception_message("Could not read the word 'false'.");
            return nullptr;
          }
          return false_value;
        }

        case 'n':
        {
          auto null_value = try_continue_read_null(++p);
          if (nullptr == null_value)
          {
            //  ERROR: could not read the word 'null'
            parse_result.assign_exception_message("Could not read the word 'null'.");
            return nullptr;
          }
          return null_value;
        }

        TJ_CASE_DIGIT
        TJ_CASE_SIGN
        {
          auto number = try_read_number(p, parse_result);
          if (nullptr == number)
          {
            //  ERROR: could not read number
            return nullptr;
          }
          return number;
        }

        TJ_CASE_BEGIN_ARRAY
        {
          // an array within an array
          parse_result.push_depth();
          auto tjvalue_array = try_continue_read_array(++p, parse_result);
          if (tjvalue_array == nullptr)
          {
            // Error: something went wrong reading an array, the error was logged.
            return nullptr;
          }
          parse_result.pop_depth();
          return tjvalue_array;
        }

        TJ_CASE_BEGIN_OBJECT
        {
          // an object within the object
          parse_result.push_depth();
          auto tjvalue_object = try_continue_read_object(++p, parse_result);
          if (tjvalue_object == nullptr)
          {
            // Error: something went wrong reading an object, the error was logged.
            return nullptr;
          }
          parse_result.pop_depth();
          return tjvalue_object;
        }

        default:
          // ERROR: Unexpected Token while trying to read value.
          parse_result.assign_exception_message("Unexpected Token while trying to read value.");
          return nullptr;
        }
      }

      // ERROR: Unable to read a string and/or value
      return nullptr;
    }

    static TJMember* try_read_string_and_value(const TJCHAR*& p, ParseResult& parse_result)
    {
      // first we look for the string, all the elements are supposed to have one.
      auto string_value = try_continue_read_string(++p, parse_result);
      if (string_value == nullptr)
      {
        //  ERROR: could not read the string
        return nullptr;
      }

      // then we look for the colon
      // only white spaces and the colon are allowed here.
      if (!try_skip_colon(p))
      {
        delete[] string_value;
        //  ERROR: could not locate the expected colon after the key value
        parse_result.assign_exception_message("Could not locate the expected colon after the key value.");
        return nullptr;
      }

      auto value = try_read_Value(p, parse_result);
      if (nullptr == value)
      {
        delete[] string_value;
        //  ERROR: Could not read the value, the error was logged.
        return nullptr;
      }
      return TJMember::move(string_value, value);
    }

    /// <summary>
    /// Return true of the content of the given source has a utf-8 bom
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    static bool has_utf8_bom(const TJCHAR* source)
    {
      if (source == nullptr)
      {
        return false;
      }

      const auto& c0 = *source;
      if (c0 != TJ_UTF8_BOM0)
      {
        return false;
      }
      const auto& c1 = *(source +1);
      if (c1 != TJ_UTF8_BOM1)
      {
        return false;
      }
      const auto& c2 = *(source +2);
      if (c2 != TJ_UTF8_BOM2)
      {
        return false;
      }

      // if we are here, then it is utf-8
      return true;
    }
  };  // Helper class

  ///////////////////////////////////////
  /// TJ

  /// <summary>
  /// Write a value to a file.
  /// </summary>
  /// <param name="file_path">The path of the file.</param>
  /// <param name="root">the value we are writting</param>
  /// <param name="write_options">The options we will be using to write</param>
  /// <returns></returns>
  bool TJ::write_file(const TJCHAR* file_path, const TJValue& root, const write_options& write_options)
  {
    return internal_write_file(file_path, root, write_options);
  }

  /// <summary>
  /// Parse a json file
  /// </summary>
  /// <param name="source">The source file we are trying to parse.</param>
  /// <param name="parse_options">The option we want to use when parsing this.</param>
  /// <returns></returns>
  TJValue* TJ::parse_file(const TJCHAR* file_path, const parse_options& parse_options)
  {
    // sanity check
    if (nullptr == file_path)
    {
      return nullptr;
    }

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
      // ERROR: Could not open the file
      return nullptr;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    TJCHAR* buffer = new TJCHAR[file_size+1];
    if (!file.read(buffer, file_size))
    {
      delete[] buffer;
      return nullptr;
    }
    buffer[file_size] = TJ_NULL_TERMINATOR;

    // we can explicitely close the file to free the resources.
    file.close();

    try
    {
      // parse the file.
      auto value = internal_parse(buffer, parse_options);

      // get rid of the buffer
      delete[] buffer;

      // return whatever we managed to read out of the file.
      return value;
    }
    catch (...)
    {
      // get rid of the buffer
      delete[] buffer;

      // rethrow.
      throw;
    }
  }

  /// <summary>
  /// Parse a json string
  /// </summary>
  /// <param name="source">The source we are trying to parse.</param>
  /// <param name="options">The option we want to use when parsing this.</param>
  /// <returns></parse_options>
  TJValue* TJ::parse(const TJCHAR* source, const parse_options& parse_options)
  {
    return internal_parse(source, parse_options);
  }

  /// <summary>
  /// Return if the given source is valid or not.
  /// </summary>
  /// <param name="source"></param>
  /// <param name="parse_options"></param>
  /// <returns></returns>
  bool TJ::is_valid(const TJCHAR* source, const parse_options& parse_options)
  {
    auto* tj_value = internal_parse(source, parse_options);
    auto is_valid = tj_value != nullptr;
    delete tj_value;
    return is_valid;
  }

  /// <summary>
  /// Internal parsing of a json source
  /// We will use the option to throw, (or not).
  /// </summary>
  /// <param name="source"></param>
  /// <param name="parse_options"></param>
  /// <returns></returns>
  TJValue* TJ::internal_parse(const TJCHAR* source, const parse_options& parse_options)
  {
    // sanity check
    ParseResult parse_result(parse_options);
    if (nullptr == source)
    {
      parse_result.assign_exception_message("The given source is null.");
      parse_result.throw_if_exception();
      return nullptr;
    }

    // if we have a utf-8 content then we just skip those.
    if (TJHelper::has_utf8_bom(source))
    {
      source += 3;
    }

    // we can only have one value and nothing else
    TJValue* value_found = nullptr;
    while (*source != TJ_NULL_TERMINATOR)
    {
      switch (*source)
      {
      TJ_CASE_SPACE
        source++;
        break;

      default:
        if (nullptr != value_found)
        {
          // Error: Unexpected multiple JSON values in root.
          parse_result.assign_exception_message("Unexpected multiple JSON values in root.");
          delete value_found;
          TJASSERT(parse_result.has_exception_message());
          parse_result.throw_if_exception();
          return nullptr;
        }

        // try and look for the value
        value_found = TJHelper::try_read_Value(source, parse_result);
        if (nullptr == value_found)
        {
          // there was an issue trying to parse.
          TJASSERT(parse_result.has_exception_message());
          parse_result.throw_if_exception();
          return nullptr;
        }
        break;
      }
    }

    if (parse_options.specification == parse_options::rfc4627 && !parse_result.has_exception_message())
    {
      if (value_found == nullptr || (!value_found->is_array() && !value_found->is_object()))
      {
        // error: RFC 4627: A JSON text must be either an object or an array.
        parse_result.assign_exception_message("RFC 4627: A JSON text must be either an object or an array.");
        delete value_found;

        // throw if the options want us to, otherwise return null.
        parse_result.throw_if_exception();
        return nullptr;
      }
    }

    // return if we found anything.
    // if we found nothing ... then it is not an error, just an empty string
    parse_result.throw_if_exception();
    return value_found != nullptr ? value_found : new TJValueString(TJCHARPREFIX(""));
  }

  /// <summary>
  /// Write a value to a file.
  /// </summary>
  /// <param name="file_path">The path of the file.</param>
  /// <param name="root">the value we are writting</param>
  /// <param name="write_options">The options we will be using to write</param>
  /// <returns></returns>
  bool TJ::internal_write_file(const TJCHAR* file_path, const TJValue& root, const write_options& write_options)
  {
    WriteResult write_result(write_options);

    //  create the json first before we open anything.
    auto json = root.dump(write_result.options().write_formating);
    if (nullptr == json)
    {
      write_result.assign_exception_message("Unable to dump the json.");
      write_result.throw_if_exception();
      return false;
    }

    //  try and optn the file...
    std::ofstream outFile(file_path, std::ios::out | std::ios::binary);
    if (!outFile)
    {
      write_result.assign_exception_message("Unable to open file for writting.");
      write_result.throw_if_exception();
      return false;
    }

    if (write_result.options().byte_order_mark == write_options::utf8)
    {
      char utf8_marker[3]{ TJ_UTF8_BOM0, TJ_UTF8_BOM1, TJ_UTF8_BOM2 };
      outFile.write(utf8_marker, 3);
      if (!outFile)
      {
        write_result.assign_exception_message("Unable to write UTF-8 BOM.");
        write_result.throw_if_exception();
        return false;
      }
    }

    outFile.write(json, strlen(json));

    if (!outFile) 
    {
      write_result.assign_exception_message("Unable to write to file.");
      write_result.throw_if_exception();
      return false;
    }

    outFile.close();

    if (!outFile) 
    {
      write_result.assign_exception_message("Unable to close the file.");
      write_result.throw_if_exception();
      return false;
    }
    return true;
  }

  ///////////////////////////////////////
  /// TJMember
  TJMember::TJMember(const TJCHAR* string, const TJValue* value):
    _string(nullptr),
    _value(nullptr)
  {
    if (string != nullptr)
    {
      const auto& length = TJHelper::string_length(string);
      _string = new TJCHAR[length + 1];
      TJHelper::copy_string(string, _string, length);
    }
    if (value != nullptr)
    {
      _value = value->clone();
    }
  }

  /// <summary>
  /// Move a value to the member
  /// </summary>
  void TJMember::move_value(TJValue*& value)
  {
    if (_value != nullptr)
    {
      delete _value;
    }
    _value = value;
    value = nullptr;
  }

  TJMember* TJMember::move(TJCHAR*& string, TJValue*& value)
  {
    auto member = new TJMember(nullptr, nullptr);
    member->_string = string;
    member->_value = value;

    value = nullptr;
    string = nullptr;
    return member;
  }

  TJMember::~TJMember()
  {
    free_string();
    free_value();
  }

  const TJCHAR* TJMember::name() const
  {
    return _string;
  }

  const TJValue* TJMember::value() const
  {
    return _value;
  }

  void TJMember::free_string()
  {
    if (nullptr != _string)
    {
      delete[] _string;
    }
    _string = nullptr;
  }

  void TJMember::free_value()
  {
    delete _value;
    _value = nullptr;
  }

  ///////////////////////////////////////
  /// TJValue
  TJValue::TJValue() : 
    _last_dump(nullptr)
  {
  }

  TJValue::~TJValue()
  {
    free_last_dump();
  }

  void TJValue::free_last_dump()  const
  {
    if (nullptr != _last_dump)
    {
      delete[] _last_dump;
      _last_dump = nullptr;
    }
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValue::clone() const
  {
    return internal_clone();
  }

  bool TJValue::is_object() const
  {
    return false;
  }

  bool TJValue::is_array() const
  {
    return false;
  }

  bool TJValue::is_string() const
  {
    return false;
  }

  bool TJValue::is_number() const
  {
    return false;
  }

  bool TJValue::is_true() const
  {
    return false;
  }

  bool TJValue::is_false() const
  {
    return false;
  }

  bool TJValue::is_null() const
  {
    return false;
  }

  const TJCHAR* TJValue::dump(formating formating, const TJCHAR* indent) const
  {
    free_last_dump();
    switch (formating)
    {
    case formating::minify:
      {
        internal_dump_configuration configuration(formating, nullptr, 
          TJCHARPREFIX(","), 
          TJCHARPREFIX(":"), 
          TJCHARPREFIX("\""), 
          TJCHARPREFIX("\""), nullptr, true);
        internal_dump(configuration, nullptr);
        _last_dump = configuration._buffer;
      }
      break;
    case formating::indented:
      {
        internal_dump_configuration configuration(formating, indent, 
          TJCHARPREFIX(","), 
          TJCHARPREFIX(": "), 
          TJCHARPREFIX("\""), 
          TJCHARPREFIX("\""), 
          TJCHARPREFIX("\n"), true);
        internal_dump(configuration, nullptr);
        _last_dump = configuration._buffer;
      }
      break;
    }
    return _last_dump;
  }

  const TJCHAR* TJValue::dump_string() const
  {
    free_last_dump();
    internal_dump_configuration configuration(formating::minify, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false);
    internal_dump(configuration, nullptr);
    _last_dump = configuration._buffer;
    return _last_dump;
  }

  bool TJValue::get_boolean(bool strict) const
  {
    const auto* boolean_object = dynamic_cast<const TJValueBoolean*>(this);
    if (nullptr != boolean_object)
    {
      return boolean_object->is_true();
    }
    if (strict)
    {
      throw TJParseException("The value is not a boolean!");
    }
    auto null_object = dynamic_cast<const TJValueNull*>(this);
    if (nullptr != null_object)
    {
      return false;
    }
    auto string_object = dynamic_cast<const TJValueString*>(this);
    if (nullptr != string_object)
    {
      throw TJParseException("String cannot be converteed to boolean!");
    }
    if (!is_number())
    {
      return false;
    }
    auto number = static_cast<const TJValueNumber*>(this);
    return number->get_number() != 0;
  }

  long double TJValue::get_float(bool strict) const
  {
    auto number_object = dynamic_cast<const TJValueNumber*>(this);
    if (nullptr != number_object)
    {
      return number_object->get_float();
    }
    if (strict)
    {
      throw TJParseException("The value is not a number!");
    }
    auto null_object = dynamic_cast<const TJValueNull*>(this);
    if (nullptr != null_object)
    {
      return 0.0;
    }
    auto string_object = dynamic_cast<const TJValueString*>(this);
    if (nullptr != string_object)
    {
      throw TJParseException("String cannot be converteed to number!");
    }
    auto boolean_object = dynamic_cast<const TJValueBoolean*>(this);
    if (nullptr != boolean_object)
    {
      return boolean_object->is_true() ? 1.0 : 0.0;
    }
    return 0.0;
  }

  long long TJValue::get_number(bool strict) const
  {
    auto number_object = dynamic_cast<const TJValueNumber*>(this);
    if (nullptr != number_object)
    {
      return number_object->get_number();
    }
    if (strict)
    {
      throw TJParseException("The value is not a number!");
    }
    auto null_object = dynamic_cast<const TJValueNull*>(this);
    if (nullptr != null_object)
    {
      return 0.0;
    }
    auto string_object = dynamic_cast<const TJValueString*>(this);
    if (nullptr != string_object)
    {
      throw TJParseException("String cannot be converteed to number!");
    }

    auto boolean_object = dynamic_cast<const TJValueBoolean*>(this);
    if (nullptr != boolean_object)
    {
      return boolean_object->is_true() ? 1 : 0;
    }
    return 0;
  }

  std::vector<long double> TJValue::get_floats(bool strict) const
  {
    auto array_object = dynamic_cast<const TJValueArray*>(this);
    if (nullptr != array_object)
    {
      return array_object->get_floats();
    }

    // not an array then so return what we have.
    return { get_float(strict) };
  }

  std::vector<long long> TJValue::get_numbers(bool strict) const
  {
    auto array_object = dynamic_cast<const TJValueArray*>(this);
    if (nullptr != array_object)
    {
      return array_object->get_numbers();
    }

    // not an array then so return what we have.
    return { get_number(strict) };
  }

  const TJCHAR* TJValue::get_string(bool strict) const
  {
    auto string_object = dynamic_cast<const TJValueString*>(this);
    if (nullptr != string_object)
    {
      return string_object->raw_value();
    }
    if (strict)
    {
      throw TJParseException("The value is not a string!");
    }
    const auto* boolean_object = dynamic_cast<const TJValueBoolean*>(this);
    if (nullptr != boolean_object)
    {
      return boolean_object->is_true() ? TJCHARPREFIX("true") : TJCHARPREFIX("false");
    }
    auto null_object = dynamic_cast<const TJValueNull*>(this);
    if (nullptr != null_object)
    {
      return TJCHARPREFIX("null");
    }

    if (is_array() || is_object())
    {
      throw TJParseException("Arrays and objects cannot be converteed to string!");
    }
    auto number = static_cast<const TJValueNumber*>(this);
    return number->dump_string();
  }

  ///////////////////////////////////////
  /// TJValue string
  TJValueString::TJValueString(const TJCHAR* value) :
    _value(nullptr)
  {
    if (value != nullptr)
    {
      const auto& length = TJHelper::string_length(value);
      _value = new TJCHAR[length+1];
      TJHelper::copy_string(value, _value, length);
    }
  }

  TJValueString::~TJValueString()
  {
    free_value();
  }

  TJValueString* TJValueString::move(TJCHAR*& value)
  {
    auto string = new TJValueString(nullptr);
    string->_value = value;
    value = nullptr;
    return string;
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueString::internal_clone() const
  {
    return new TJValueString(_value);
  }

  void TJValueString::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    //  unused
    (void)current_indent;

    // add the quote, (if we have one)
    TJHelper::add_string_to_string(configuration._value_quote, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
    if (nullptr == _value)
    {
      TJHelper::add_string_to_string(TJCHARPREFIX(""), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
    }
    else if (configuration._escape_special_characters)
    {
      const TJCHAR* p = _value;
      while (*p != TJ_NULL_TERMINATOR)
      {
        switch (*p)
        {
        case TJ_ESCAPE_QUOTATION: // % x22 / ; "    quotation mark  U+0022
          TJHelper::add_string_to_string(TJCHARPREFIX("\\\""), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_REVERSE_SOLIDUS: // % x5C / ; \    reverse solidus U + 005C
          TJHelper::add_string_to_string(TJCHARPREFIX("\\\\"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_SOLIDUS: // % x2F / ; / solidus         U + 002F
          TJHelper::add_string_to_string(TJCHARPREFIX("\\/"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_BACKSPACE: // % x62 / ; b    backspace       U + 0008
          TJHelper::add_string_to_string(TJCHARPREFIX("\\b"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_FORM_FEED: // % x66 / ; f    form feed       U + 000C
          TJHelper::add_string_to_string(TJCHARPREFIX("\\f"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_LINE_FEED:  // % x6E / ; n    line feed       U + 000A
          TJHelper::add_string_to_string(TJCHARPREFIX("\\n"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_CARRIAGE_RETURN:  // % x72 / ; r    carriage return U + 000D
          TJHelper::add_string_to_string(TJCHARPREFIX("\\r"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        case TJ_ESCAPE_TAB:  // % x74 / ; t    tab             U + 0009
          TJHelper::add_string_to_string(TJCHARPREFIX("\\t"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;

        default:
          TJHelper::add_char_to_string(*p, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
          break;
        }
        ++p;
      }
    }
    else
    {
      TJHelper::add_string_to_string(_value, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
    }

    // then close the quote, (if we have one)
    TJHelper::add_string_to_string(configuration._value_quote, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  bool TJValueString::is_string() const
  {
    return true;
  }

  void TJValueString::free_value()
  {
    if (nullptr != _value)
    {
      delete[] _value;
    }
    _value = nullptr;
  }

  const TJCHAR* TJValueString::raw_value() const
  {
    return _value == nullptr ? TJCHARPREFIX("") : _value;
  }


  ///////////////////////////////////////
  /// TJValue true
  TJValueBoolean::TJValueBoolean(bool is_true) :
    _is_true(is_true)
  {
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueBoolean::internal_clone() const
  {
    return new TJValueBoolean(_is_true);
  }

  void TJValueBoolean::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    //  unused
    (void)current_indent;

    // then the word we are after
    TJHelper::add_string_to_string(_is_true ? TJCHARPREFIX("true"): TJCHARPREFIX("false"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  bool TJValueBoolean::is_true() const
  {
    return _is_true;
  }

  bool TJValueBoolean::is_false() const
  {
    return !_is_true;
  }

  ///////////////////////////////////////
  /// TJValue null
  TJValueNull::TJValueNull()
  {
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueNull::internal_clone() const
  {
    return new TJValueNull();
  }

  void TJValueNull::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    //  unused
    (void)current_indent;

    // then the word we are after
    TJHelper::add_string_to_string(TJCHARPREFIX("null"), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  bool TJValueNull::is_null() const
  {
    return true;
  }

  ///////////////////////////////////////
  /// TJValueObject
  TJValueObject::TJValueObject() :
    _members(nullptr)
  {
  }

  TJValueObject::~TJValueObject()
  {
    free_members();
  }

  /// <summary>
  /// Pop a value out of the list of items
  /// </summary>
  /// <param name="key"></param>
  void TJValueObject::pop(const TJCHAR* key)
  {
    if (nullptr == _members)
    {
      return;
    }
#if TJ_INCLUDE_STDVECTOR == 1
    auto it = std::find_if(_members->begin(), _members->end(), [&](TJMember* value) {
      return TJHelper::are_same(key, value->name(), true);
      });
    if (it != _members->end())
    {
      delete* it;
      _members->erase(it);
    }
#else
    _members->pop(key);
#endif
  }

  /// <summary>
  /// Set the value of a string.
  /// </summary>
  /// <param name="key"></param>
  /// <param name="value"></param>
  /// <returns></returns>
  void TJValueObject::set_string(const TJCHAR* key, const char* value)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_string = new TJValueString(value);
    member->move_value(value_string);
    TJHelper::move_member_to_members(member, _members);
  }

  /// <summary>
  /// Set the value to null.
  /// </summary>
  /// <param name="key"></param>
  /// <returns></returns>
  void TJValueObject::set_null(const TJCHAR* key)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_null = new TJValueNull();
    member->move_value(value_null);
    TJHelper::move_member_to_members(member, _members);
  }

  /// <summary>
  /// Set the value of a ... value
  /// </summary>
  /// <param name="key"></param>
  /// <param name="value"></param>
  /// <returns></returns>
  void TJValueObject::set(const TJCHAR* key, const TJValue* value)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    auto clone = value->clone();
    member->move_value(clone);
    TJHelper::move_member_to_members(member, _members);
  }

  /// <summary>
  /// Set the value of a number
  /// </summary>
  /// <param name="key"></param>
  /// <param name="value"></param>
  /// <returns></returns>
  void TJValueObject::set_number(const TJCHAR* key, long long value)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_int = new TJValueNumberInt(value);
    member->move_value(value_int);
    TJHelper::move_member_to_members(member, _members);
  }

  /// <summary>
  /// Set the value of a number
  /// </summary>
  /// <param name="key"></param>
  /// <param name="value"></param>
  /// <returns></returns>
  void TJValueObject::set_float(const TJCHAR* key, long double value)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_number = TJHelper::try_create_number_from_float(value);
    member->move_value(value_number);
    TJHelper::move_member_to_members(member, _members);
  }

  /// <summary>
  /// Set the value a boolean
  /// </summary>
  /// <param name="key"></param>
  /// <param name="value"></param>
  /// <returns></returns>
  void TJValueObject::set_boolean(const TJCHAR* key, bool value)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_boolean = new TJValueBoolean(value);
    member->move_value(value_boolean);
    TJHelper::move_member_to_members(member, _members);
  }

  TJValueObject* TJValueObject::move(TJDICTIONARY*& members)
  {
    auto object = new TJValueObject();
    object->_members = members;
    members = nullptr;
    return object;
  }

  long double TJValueObject::get_float(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if(throw_if_not_found)
      { 
        throw TJParseException("The key was not found!");
      }
      return 0.0;
    }
    return value->get_float();
  }

  long long TJValueObject::get_number(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if (throw_if_not_found)
      {
        throw TJParseException("The key was not found!");
      }
      return 0.0;
    }
    return value->get_number();
  }

  std::vector<long double> TJValueObject::get_floats(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if (throw_if_not_found)
      {
        throw TJParseException("The key was not found!");
      }
      return {};
    }
    return value->get_floats();
  }

  std::vector<long long> TJValueObject::get_numbers(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if (throw_if_not_found)
      {
        throw TJParseException("The key was not found!");
      }
      return {};
    }
    return value->get_numbers();
  }

  const TJCHAR* TJValueObject::get_string(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if (throw_if_not_found)
      {
        throw TJParseException("The key was not found!");
      }
      return TJCHARPREFIX("");
    }
    return value->get_string();
  }

  bool TJValueObject::get_boolean(const TJCHAR* key, bool case_sensitive, bool throw_if_not_found) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      if (throw_if_not_found)
      {
        throw TJParseException("The key was not found!");
      }
      return false;
    }
    return value->get_boolean();
  }

  void TJValueObject::set_floats(const TJCHAR* key, const std::vector<long double>& values)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_array = new TJValueArray();
    auto array = reinterpret_cast<TJValueArray*>(value_array);
    for(const auto& value : values)
    {
      array->add_float(value);
    }
    member->move_value(value_array);
    TJHelper::move_member_to_members(member, _members);
  }

  void TJValueObject::set_numbers(const TJCHAR* key, const std::vector<long long>& values)
  {
    if (nullptr == _members)
    {
      _members = new TJDICTIONARY();
    }

    auto member = new TJMember(key, nullptr);
    TJValue* value_array = new TJValueArray();
    auto array = reinterpret_cast<TJValueArray*>(value_array);
    for (const auto& value : values)
    {
      array->add_number(value);
    }
    member->move_value(value_array);
    TJHelper::move_member_to_members(member, _members);
  }


  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueObject::internal_clone() const
  {
    auto object = new TJValueObject();
    if (_members != nullptr)
    {
      auto members = new TJDICTIONARY();
#if TJ_INCLUDE_STDVECTOR == 1
      for (auto& member : *_members)
      {
        members->push_back(new TJMember(member->name(), member->value()->clone()));
      }
#else
      auto size = _members->size();
      for(unsigned int i = 0; i < size; ++i)
      {
        const auto& member = _members->at(i);
        const auto& name = member->name();
        members->set(new TJMember(name, member->value()->clone()));
      }
#endif
      object->_members = members;
    }
    return object;
  }

  void TJValueObject::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    // open it
    TJHelper::add_char_to_string('{', configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

    auto number_of_items = get_number_of_items();
    if (number_of_items > 0)
    {
      // only return if we have data.
      if (configuration._formating == formating::indented)
      {
        TJHelper::add_char_to_string(TJ_ESCAPE_LINE_FEED, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
      }

      int inner_buffer_pos = 0;
      int inner_buffer_max_length = 0;
      TJCHAR* inner_current_indent = nullptr;

      TJHelper::add_string_to_string(current_indent, inner_current_indent, inner_buffer_pos, inner_buffer_max_length);
      TJHelper::add_string_to_string(configuration._indent, inner_current_indent, inner_buffer_pos, inner_buffer_max_length);

#if TJ_INCLUDE_STDVECTOR == 1
      for (const auto& member : *_members)
      {
#else
      auto size = _members->size();
      for (unsigned int i = 0; i < size; ++i)
      {
        const auto& member = _members->at(i);
#endif
        TJHelper::add_string_to_string(inner_current_indent, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

        TJHelper::add_string_to_string(configuration._key_quote, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        TJHelper::add_string_to_string(member->name(), configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        TJHelper::add_string_to_string(configuration._key_quote, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

        TJHelper::add_string_to_string(configuration._key_separator, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

        member->value()->internal_dump(configuration, inner_current_indent);

        // don't add on the last item
        if (--number_of_items > 0)
        {
          TJHelper::add_string_to_string(configuration._item_separator, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        }
        if (configuration._formating == formating::indented)
        {
          TJHelper::add_char_to_string(TJ_ESCAPE_LINE_FEED, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        }
      }
      delete[] inner_current_indent;
    }
    // close it.
    TJHelper::add_string_to_string(current_indent, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
    TJHelper::add_char_to_string('}', configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  bool TJValueObject::is_object() const
  {
    return true;
  }

  unsigned int TJValueObject::get_number_of_items() const
  {
    return _members == nullptr ? 0 : _members->size();
  }

  TJMember* TJValueObject::operator [](int idx) const
  {
    return at(idx);
  }

  TJMember* TJValueObject::at(int idx) const
  {
    if(idx < 0 || unsigned(idx) >= get_number_of_items())
    {
      return nullptr;
    }
#if TJ_INCLUDE_STDVECTOR == 1
    return (*_members)[idx];
#else
    return _members->at(idx);
#endif
  }

  void TJValueObject::free_members()
  {
    if (_members == nullptr)
    {
      return;
    }

#if TJ_INCLUDE_STDVECTOR == 1
    for (auto var : *_members)
    {
      delete var;
    }
#endif
    delete _members;
    _members = nullptr;
  }

  /// <summary>
  /// Try and get the value of this member if it exists.
  /// </summary>
  /// <param name="key"></param>
  /// <returns></returns>
  const TJValue* TJValueObject::try_get_value(const TJCHAR* key, bool case_sensitive) const
  {
    if (nullptr == key)
    {
      return nullptr;
    }
    if (nullptr == _members)
    {
      return nullptr;
    }

#if TJ_INCLUDE_STDVECTOR == 1
    auto it = std::find_if(_members->begin(), _members->end(), [&](TJMember* value) {
      return TJHelper::are_same(key, value->name(), case_sensitive);
    });

    return (it == _members->end()) ? nullptr : (*it)->value();
#else
    auto member = _members->at(key, case_sensitive);
    if (nullptr == member)
    {
      return nullptr;
    }
    return member->value();
#endif
  }

  /// <summary>
  /// Try and get a string value, if it does not exist, then we return null.
  /// </summary>
  /// <param name="key"></param>
  /// <returns></returns>
  const TJCHAR* TJValueObject::try_get_string(const TJCHAR* key, bool case_sensitive) const
  {
    auto value = try_get_value(key, case_sensitive);
    if (nullptr == value)
    {
      return nullptr;
    }

    delete [] value->_last_dump;
    internal_dump_configuration configuration(formating::minify, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, false);
    value->internal_dump(configuration, nullptr);
    value->_last_dump = configuration._buffer;

    return value->_last_dump;
  }

  ///////////////////////////////////////
  /// TJValueArray
  TJValueArray::TJValueArray() :
    _values(nullptr)
  {
  }

  TJValueArray::~TJValueArray()
  {
    free_values();
  }

  TJValueArray* TJValueArray::move(TJLIST*& values)
  {
    auto value = new TJValueArray();
    value->_values = values;
    values = nullptr;
    return value;
  }

  void TJValueArray::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    // open it
    TJHelper::add_char_to_string('[', configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

    auto number_of_items = get_number_of_items();
    if (number_of_items > 0)
    {
      // only return if we have data.
      TJHelper::add_string_to_string(configuration._new_line, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

      int inner_buffer_pos = 0;
      int inner_buffer_max_length = 0;
      TJCHAR* inner_current_indent = nullptr;
      
      TJHelper::add_string_to_string(current_indent, inner_current_indent, inner_buffer_pos, inner_buffer_max_length);
      TJHelper::add_string_to_string(configuration._indent, inner_current_indent, inner_buffer_pos, inner_buffer_max_length);

#if TJ_INCLUDE_STDVECTOR == 1
      for (const auto& value : *_values)
      {
#else
      auto size = _values->size();
      for (unsigned int i = 0; i < size; ++i)
      {
        const auto& value = _values->at(i);
#endif
        TJHelper::add_string_to_string(inner_current_indent, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        value->internal_dump(configuration, inner_current_indent);

        // don't add on the last item
        if (--number_of_items > 0)
        {
          TJHelper::add_string_to_string(configuration._item_separator, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
        }
        TJHelper::add_string_to_string(configuration._new_line, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
      }
      delete[] inner_current_indent;
    }
    // close it.
    TJHelper::add_string_to_string(current_indent, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
    TJHelper::add_char_to_string(']', configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueArray::internal_clone() const
  {
    auto array = new TJValueArray();
    if (_values != nullptr)
    {
      auto values = new TJLIST();
#if TJ_INCLUDE_STDVECTOR == 1
      for (const auto& value : *_values)
      {
        values->push_back(value->clone());
#else
      auto size = _values->size();
      for (unsigned int i = 0; i < size; ++i)
      {
        const auto& value = _values->at(i);
        values->add(value->clone());
#endif
      }
      array->_values = values;
    }
    return array;
  }

  bool TJValueArray::is_array() const
  {
    return true;
  }

  unsigned int TJValueArray::get_number_of_items() const
  {
    return _values == nullptr ? 0 : _values->size();
  }

  TJValue* TJValueArray::operator [](int idx) const
  {
    return at(idx);
  }

  TJValue* TJValueArray::at(int idx) const
  {
    if (idx < 0 || unsigned(idx) >= get_number_of_items())
    {
      return nullptr;
    }
#if TJ_INCLUDE_STDVECTOR == 1
    return (*_values)[idx];
#else
    return _values->at(idx);
#endif
  }

  void TJValueArray::free_values()
  {
    if (_values == nullptr)
    {
      return;
    }

#if TJ_INCLUDE_STDVECTOR == 1
    for (auto var : *_values)
    {
      delete var;
    }
#endif
    delete _values;
    _values = nullptr;
  }

  std::vector<long double> TJValueArray::get_floats(bool throw_if_not_numbers) const
  {
    std::vector<long double> values = {};
    for( unsigned int i = 0; i < get_number_of_items(); ++i)
    {
      auto value = _values->at(i);
      if (!value->is_number())
      {
        if (throw_if_not_numbers)
        {
          throw TJParseException("One or more values in the array is not a number!");
        }
        continue;
      }
      auto number = static_cast<TJValueNumber*>(value);
      values.push_back(number->get_float());
    }
    return values;
  }

  std::vector<long long> TJValueArray::get_numbers(bool throw_if_not_numbers) const
  {
    std::vector<long long> values = {};
    for (unsigned int i = 0; i < get_number_of_items(); ++i)
    {
      auto value = _values->at(i);
      if (!value->is_number())
      {
        if (throw_if_not_numbers)
        {
          throw TJParseException("One or more values in the array is not a number!");
        }
        continue;
      }
      auto number = static_cast<TJValueNumber*>(value);
      values.push_back(number->get_number());
    }
    return values;
  }

  void TJValueArray::add(const TJValue* value)
  {
    if (nullptr == value)
    {
      auto* nullObject = new TJValueNull();
      add(nullObject);
      delete nullObject;
      return;
    }

    if (_values == nullptr)
    {
      _values = new TJLIST();
    }
#if TJ_INCLUDE_STDVECTOR == 1
    _values->push_back(value->clone());
#else
    _values->add(value->clone());
#endif
  }

  void TJValueArray::add_numbers(const std::vector<long long>& values)
  {
    for (const auto& value : values)
    {
      add_number(value);
    }
  }

  void TJValueArray::add_numbers(const std::vector<long>& values)
  {
    for (const auto& value : values)
    {
      add_number(value);
    }
  }

  void TJValueArray::add_numbers(const std::vector<int>& values)
  {
    for (const auto& value : values)
    {
      add_number(value);
    }
  }

  void TJValueArray::add_floats(const std::vector<long double>& values)
  {
    for (const auto& value : values)
    {
      add_float(value);
    }
  }

  void TJValueArray::add_floats(const std::vector<double>& values)
  {
    for (const auto& value : values)
    {
      add_float(value);
    }
  }

  void TJValueArray::add_floats(const std::vector<float>& values)
  {
    for (const auto& value : values)
    {
      add_float(value);
    }
  }

  void TJValueArray::add_number(long long value)
  {
    auto* objectNumber = new TJValueNumberInt(value);
    add(objectNumber);
    delete objectNumber;
  }
  
  void TJValueArray::add_float(long double value)
  {
    auto* tjNumber = TJHelper::try_create_number_from_float(value);
    add(tjNumber);
    delete tjNumber;
  }
  
  void TJValueArray::add_boolean(bool value)
  {
    auto* objectBoolean = new TJValueBoolean(value);
    add(objectBoolean);
    delete objectBoolean;
  }

  void TJValueArray::add_string(const char* value)
  {
    auto* objectString = new TJValueString(value);
    add(objectString);
    delete objectString;
  }

  ///////////////////////////////////////
  /// TJValue Number
  TJValueNumber::TJValueNumber(const bool is_negative) : 
    _is_negative(is_negative)
  {
  }

  bool TJValueNumber::is_number() const
  {
    return true;
  }

  long double TJValueNumber::get_float() const
  {
    auto value_float = dynamic_cast<const TinyJSON::TJValueNumberFloat*>(this);
    if (nullptr != value_float)
    {
      return value_float->get_number();
    }

    auto value_int = dynamic_cast<const TinyJSON::TJValueNumberInt*>(this);
    if (nullptr != value_int)
    {
      return static_cast<long double>(value_int->get_number());
    }

    auto value_exponent = dynamic_cast<const TinyJSON::TJValueNumberExponent*>(this);
    if (nullptr != value_exponent)
    {
      // probably over/underflow
      return static_cast<long double>(value_exponent->get_number());
    }

    throw TJParseException("The value is not a number!");
  }

  long long TJValueNumber::get_number() const
  {
    auto value_float = dynamic_cast<const TinyJSON::TJValueNumberFloat*>(this);
    if (nullptr != value_float)
    {
      return static_cast<long long>(value_float->get_number());
    }

    auto value_int = dynamic_cast<const TinyJSON::TJValueNumberInt*>(this);
    if (nullptr != value_int)
    {
      return value_int->get_number();
    }

    auto value_exponent = dynamic_cast<const TinyJSON::TJValueNumberExponent*>(this);
    if (nullptr != value_exponent)
    {
      // probably over/underflow
      return static_cast<long long>(value_exponent->get_number());
    }

    throw TJParseException("The value is not a number!");
  }


  ///////////////////////////////////////
  /// TJValue whole Number
  TJValueNumberInt::TJValueNumberInt(const unsigned long long& number, const bool is_negative) :
    TJValueNumber(is_negative),
    _number(number)
  {
  }

  TJValueNumberInt::TJValueNumberInt(const long long& number) :
    TJValueNumber(number<0),
    _number(number < 0 ? -1*number : number)
  {
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueNumberInt::internal_clone() const
  {
    return new TJValueNumberInt(_number, _is_negative);
  }

  void TJValueNumberInt::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    //  unused
    (void)current_indent;

    // if we have no fraction, then just return it.
    auto string = TJHelper::fast_number_to_string(_number, 0, _is_negative );

    // then the number
    TJHelper::add_string_to_string(string, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);

    delete[] string;
  }

  long long TJValueNumberInt::get_number() const
  {
    return _is_negative ? -1* _number : _number;
  }

  ///////////////////////////////////////
  /// TJValue float Number
  TJValueNumberFloat::TJValueNumberFloat(const unsigned long long& number, const unsigned long long& fraction, const unsigned int& fraction_exponent, bool is_negative) :
    TJValueNumber(is_negative),
    _string(nullptr),
    _number(number),
    _fraction(fraction),
    _fraction_exponent(fraction_exponent)
  {
  }

  TJValueNumberFloat::TJValueNumberFloat(long double number) :
    TJValueNumber(number<0),
    _string(nullptr),
    _number(TJHelper::get_whole_number_from_float(number)),
    _fraction(TJHelper::get_fraction_from_float(number)),
    _fraction_exponent(TJHelper::get_unsigned_exponent_from_float(number))
  {
  }

  TJValueNumberFloat::~TJValueNumberFloat()
  {
    if (nullptr != _string)
    {
      delete[] _string;
    }
  }

  void TJValueNumberFloat::make_string_if_needed() const
  {
    if (nullptr != _string)
    {
      return;
    }

    _string = TJHelper::fast_number_and_fraction_to_string(_number, _fraction, _fraction_exponent, _is_negative);
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueNumberFloat::internal_clone() const
  {
    return new TJValueNumberFloat(_number, _fraction, _fraction_exponent, _is_negative);
  }

  void TJValueNumberFloat::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    // unused
    (void)current_indent;

    // make sthe string is needed
    make_string_if_needed();

    // then the number
    TJHelper::add_string_to_string(_string, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  long double TJValueNumberFloat::get_number() const
  {
    if (_fraction == 0) {
      return static_cast<long double>(_number);
    }

    // Convert b to its fractional form
    auto pow = static_cast<long double>(TJHelper::fast_power_of_10(_fraction_exponent));
    const auto& whole_number = _number * pow + _fraction;

    // Combine the number and the fraction
    return (_is_negative ? -1 : 1) * (whole_number / pow);
  }

  ///////////////////////////////////////
  /// TJValue float Number
  TJValueNumberExponent::TJValueNumberExponent(const unsigned long long& number, const unsigned long long& fraction, const unsigned int& fraction_exponent, const int& exponent, bool is_negative) :
    TJValueNumber(is_negative),
    _string(nullptr),
    _number(number),
    _fraction(fraction),
    _fraction_exponent(fraction_exponent),
    _exponent(exponent)    
  {
  }

  TJValueNumberExponent::~TJValueNumberExponent()
  {
    if (_string != nullptr)
    {
      delete[] _string;
    }
  }

  long double TJValueNumberExponent::get_number() const
  {
    // Convert the fractional part to a long double, properly scaled
    long double fractional = static_cast<long double>(_fraction);
    fractional /= std::pow(10.0L, _fraction_exponent);

    // Combine whole and fraction
    long double value = static_cast<long double>(_number) + fractional;
    if (_is_negative)
    {
      value *= -1.0L;
    }

    // Apply the exponent
    value *= std::pow(10.0L, _exponent);

    // Optional: check for overflow/underflow
    if (value > std::numeric_limits<long double>::max())
    {
      return std::numeric_limits<long double>::infinity();
    }
    if (value != 0.0L && std::abs(value) < std::numeric_limits<long double>::min())
    {
      return 0.0L;
    }
    return value;
  }

  /// <summary>
  /// Allow each derived class to create a copy of itself.
  /// </summary>
  /// <returns></returns>
  TJValue* TJValueNumberExponent::internal_clone() const
  {
    return new TJValueNumberExponent(_number, _fraction, _fraction_exponent, _exponent, _is_negative);
  }

  void TJValueNumberExponent::internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const
  {
    // unused
    (void)current_indent;

    // we only create the string value when the caller asks for it.
    // this is to make sure that we do not create it on parsing.
    make_string_if_needed();

    // then the number
    TJHelper::add_string_to_string(_string, configuration._buffer, configuration._buffer_pos, configuration._buffer_max_length);
  }

  void TJValueNumberExponent::make_string_if_needed() const
  {
    if (nullptr != _string)
    {
      return;
    }
    _string = TJHelper::fast_number_fraction_and_exponent_to_string(_number, _fraction, _fraction_exponent, _exponent, _is_negative);
  }
} // TinyJSON